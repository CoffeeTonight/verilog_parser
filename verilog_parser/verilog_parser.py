#!/usr/bin/env python3

__author__ = "dyxn.tonightcoffee@gmail.com"
__authors__ = ["dyxn.tonightcoffee@gmail.com"]
__contact__ = "dyxn.tonightcoffee@gmail.com"
__copyright__ = "Copyright 2023, free"
__credits__ = ["dyxn.tonightcoffee@gmail.com"]
__date__ = "2023/04/30"
__deprecated__ = False
__email__ =  "dyxn.tonightcoffee@gmail.com"
__license__ = "MIT"
__maintainer__ = "developer"
__status__ = "Testing"
__version__ = "0.0.1"


import re
import json
import os
import copy
from collections import OrderedDict


class HDL:
    def __repr__(self):
        return f'{self.name}'

    def __init__(self, name):
        self.name = name
        for i in ["inst", "port", "comment", "preprocessor", "subroutine", "param", "lparam",
                  "wire", "reg", "assign", "definition", "inlinedefine", "unknown"]:
            self.__setattr__(i, {})


class PORT:
    def __repr__(self):
        return f'{self.vlnv}'

    def __init__(self, name):
        self.name = name
        for i in ["dir", "dimm", "array", "type", "vlnv", "logicalName", "prefix", "postfix"]:
            self.__setattr__(i, str)


class HDLFILE:
    def __repr__(self):
        return f'{self.name}'

    def __init__(self, vfile):
        self.name = vfile
        self.preprocessor = {}
        self.unknown = {}
        self.inst = []


class verilog_parser():
    def __repr__(self):
        return f'{self.args.src}'

    def __init__(self, args):
        startsends = {"comment": {"//": "\n", "/*": "*/"},
                      "snippet": {"begin": "end", None: "\n"},
                      "module": {"module": "endmodule"},
                      "port": {"input": ";", "output": ";", "inout": ";"},
                      "definition": {"assign": ";", "wire": ";", "reg": ";"},
                      "rtl": {"always": "\n", "always@": "\n", "always@(": "\n"},
                      "tbd": {None: "\n"},
                      "ifdef": [
                          {"`ifdef": True, "`ifndef": False},
                          {"`else": None, "`elsif": True},
                          {"`endif": None}
                      ],
                      "preprocess": {"`define": "\n", "`timescale": "\n", "`include": "\n"},
                      "param": {"parameter": [",", ")"]},
                      "lparam": {"localparam": ";"},
                      "div": ["\n", " "]
                      }
        self.args = args
        self.filelist = []
        self.inlinedefine = {}
        self.openvfiles()
        self.hdlfile = []
        [self.hdlfile.append(HDLFILE(i)) for i in self.filelist]
        self.startsends = startsends
        self.updateDefine()
        self.seq_parse()

    def openvfiles(self):
        assert os.path.isfile(self.args.src), f"ERR:: {self.args.src} File not founded"
        if self.args.src.endswith(".v") or self.args.src.endswith(".sv"):
            self.filelist = [self.args.src]
        else:
            with open(self.args.src, "r", encoding="utf8") as f:
                self.filelist = [f"{os.path.abspath(i.strip().rstrip())}"
                                 for i in f.readlines() if i.strip().rstrip().isfile()]

    def updateDefine(self):
        defines = self.args.define
        if os.path.isfile(self.args.define):
            f = open(self.args.define, "r", encoding="utf8").readlines()
            f = [i for i in f if not (i.startswith("//") or i.startswith("#"))]
            f = {i.split("=")[0]: i.split("=")[-1] if "=" in i else True for i in "".join(f).split(",")}
            defines = f
        else:
            defines = {i.split("=")[0]: i.split("=")[-1] if "=" in i else True for i in defines.split(",")}
        self.define = defines
        self.define.update({1: True, 0: False, "1": True, "0": False})

    def cutout_code(self, txt, conts: dict):
        code = txt
        idx_ = True
        cutout_code = []
        while idx_:
            idx_ = {code.index(i): i for i in conts if i in code}
            for k in sorted(list(idx_)):
                v = idx_[k]
                idx_end = code[k:].index(conts[v])
                cutout_code += [code[k:k + idx_end + conts[v].__len__()]]
                code = code[:k] + code[k + idx_end + conts[v].__len__():]
                break
        return code, cutout_code

    def find_definition(self, txt, cond):
        ncond = {}
        for indef in re.finditer("`define\s+\w+[^\n\r]+\n", txt):
            defcode = [i for i in txt[indef.start() + "`define".__len__():
                                         indef.end()].strip().rstrip().split(" ") if i != ""]
            defcode = [defcode[0], '1'] if defcode.__len__() == 1 else [defcode[0], " ".join(defcode[1:])]
            if isinstance(defcode[-1], str) and not defcode[-1].isdigit():
                lvd = [i for i in re.findall(r'[^(][_a-zA-Z][_a-zA-Z0-9]*', " ".join(defcode[1:])) if "'" not in i]
                for i in lvd:
                    if i in cond:
                        defcode[-1] = defcode[-1].replace(i, str(int(cond[i])) \
                            if isinstance(cond[i], bool) else str(cond[i]))
            if defcode[-1].isdigit():
                valdef = int(defcode[-1])
            elif any([i in re.sub("^[0-9]*", "", defcode[-1]) for i in ["'h", "'b", "'d"]]):
                valdef = defcode[-1]
            elif defcode[-1] in cond:
                valdef = cond[defcode[-1]]
            else:
                valdef = 0
            ncond.update({defcode[0]: valdef})
            cond.update(ncond)
        return ncond


    def get_recursive(self, req, TOP=True):
        recur = [[[]]]
        for n, i in enumerate(req):
            if "Done" in i[-1]:
                continue
            if i[0] == 0:
                if 2 in recur[-1][-1] or recur[-1][-1] == []:
                    recur.append([{0: i[-1]}])
                    i[-1].update({"Done": True})
                else:
                    rrecur = self.get_recursive(req[n:], TOP=False)
                    recur[-1].append(rrecur)
            else:
                recur[-1].append({i[0]: i[-1]})
                i[-1].update({"Done": True})
                if not TOP and i[0] == 2:
                    return recur[1:]
        return recur[1:]

    def txt_recursive(self, txt, seq, condition, TOP=True, TXTPOS=0):
        rctxt = txt
        cond = condition
        for useq in seq:
            ll = useq[0]
            for k, v in ll.items():
                if isinstance(k, str):
                    v[k] = 1
                    continue
                defstart = list(v)[0]
                seqval = v[defstart][0]
                subseq_pos = {list(i)[0]: n for n, i in enumerate(useq) if isinstance(i, dict)}
                # todo: find inlinedefine & add
                curtxt = rctxt[TXTPOS:defstart]
                TXTPOS = defstart
                founded_cond = self.find_definition(curtxt, cond)
                if founded_cond != {}:
                    self.inlinedefine.update(founded_cond)
                    cond.update(founded_cond)
                # todo: ifdef again with inlinedefine
                defval = seqval["defcode"].rstrip().split(" ")[-1]
                seqval["DEF"] = (defval in cond and cond[defval]) == seqval["TF"]
                idx_next_seq = subseq_pos[1] if 1 in subseq_pos else subseq_pos[2]
                if seqval["DEF"]:
                    codestart = seqval["E"]
                    subseq_range = useq[1: idx_next_seq+1]
                else:
                    codestart = self.findValbyKey(useq[idx_next_seq], "E")
                    subseq_range = useq[idx_next_seq + 1:]
                seqend = self.findValbyKey(useq[-1], "E")
                for _s in subseq_range:
                    if isinstance(_s, list):
                        rctxt, cond = self.txt_recursive(rctxt, _s, cond, TOP=False, TXTPOS=TXTPOS)
                        continue
                    beforetxt = rctxt[defstart:seqend]
                    codeend = list(_s[list(_s)[0]])[0]
                    aftertxt = rctxt[codestart:codeend]
                    aftertxt = "\t"*(beforetxt.__len__() - aftertxt.__len__()) + aftertxt
                    assert aftertxt.__len__() == beforetxt.__len__(), "ERR:: Check Tool"
                    chgtxt = rctxt[:defstart] + aftertxt + rctxt[seqend:]
                    assert rctxt.__len__() == chgtxt.__len__(), "ERR:: Check Tool"
                    rctxt = chgtxt
                    TXTPOS = seqend
                a = rctxt[k:seqval["E"]]
        return rctxt, cond

    def findValbyKey(self, _s, key):
        if isinstance(_s, dict):
            if key in _s:
                return _s[key]
            else:
                for k, v in _s.items():
                    val = self.findValbyKey(v, key)
                    if val:
                        return val
        else:
            return None

    def merge_nestedcode(self, txt, seq, offset=0):
        cseq = [[list(v)[0], {k: v}] for k, v in seq.items()]
        seq_rec = self.get_recursive(cseq)
        cond = copy.deepcopy(self.define)
        ntxt = txt
        rc_txt, rc_cond = self.txt_recursive(ntxt, seq_rec, cond)
        return rc_txt, rc_cond

    def conditionGen(self, txt, conts, cond):
        ntxt = txt
        rgx = {}
        for n, i in enumerate(conts):
            rgx[n] = {f"{k}\s+\w+\s+" if v != None else f"{k}":v for k, v in i.items()}
        idx_rgx = {i: {} for i in range (0, n+1)}
        for n, i in rgx.items():
            for ik, iv in i.items():
                for rei in re.finditer(ik, ntxt):
                    val_DEF = True
                    if iv != None:
                        defcode = ntxt[rei.start():rei.end()].rstrip().split(" ")[-1]
                        val_DEF = (defcode in cond and int(cond[defcode])) == iv
                    idx_rgx[n][rei.start()] = {"E":rei.end(), "TF": iv, "DEF": val_DEF,
                                               "defcode": ntxt[rei.start():rei.end()]}
        nested_seq = {}
        for k, v in idx_rgx.items():
            for kk, vv in v.items():
                nested_seq[kk] = k
        assert idx_rgx[0].__len__() == idx_rgx[2].__len__(), "ERR::# ifdef != # endif. Plz Check HDL."
        nested_seq = {i:nested_seq[i] for i in sorted(list(nested_seq))}
        nested_seq = {i:{j: idx_rgx[j][i]} for i, j in nested_seq.items()}
        return ntxt, nested_seq

    def getNameWith(self, v, statement):
        _ = re.findall(f"{statement}\s+\w+", v)
        if not _:
            return None
        return _[0][statement.__len__():].strip()


    def extractFlist(self, file):
        if not (file.name.endswith(".v") or file.name.endswith(".sv")):
            flist = open(file.name, "r", encoding="utf8").readlines()
            flist = [self.extractFlist(i.strip().rstrip()) for i in flist if not (i.startswith("//") or i.startswith("#"))]
            return flist
        else:
            return file if isinstance(file, HDLFILE) else HDLFILE(file)


    def seq_parse(self):
        for file in copy.deepcopy(self.hdlfile[1:]):
            self.hdlfile += [self.extractFlist(file)]
        for file in self.hdlfile:
            hdl = open(file.name, "r", encoding="utf8").read().replace("\t", "    ")

            #todo: find module
            ncode, file.comment = self.cutout_code(hdl, self.startsends["comment"])

            # todo: handles ifdef
            ncode, file.definition = self.conditionGen(ncode, self.startsends["ifdef"], self.define)
            defcode, code_seqs = self.merge_nestedcode(ncode, file.definition)
            assert ncode.__len__() == defcode.__len__(), "ERR::Check Tool"

            outofcode, lmodule = self.cutout_code(defcode.replace("\t", " ").replace("\\\n", " "), self.startsends["module"])

            for v in outofcode.replace("\t", "").split("\n"):
                v = v.strip().rstrip()
                if v == "":
                    continue
                #grep preprocessor
                if any([v.startswith(i) for i in list(self.startsends["preprocess"])]):
                    if "inCommand" not in file.preprocessor:
                        file.preprocessor.update({"inCommand": self.define, "code": []})
                    file.preprocessor["code"] += [v]
                else:
                    if "outofcode" not in file.unknown:
                        file.unknown.update({"outofcode": []})
                    file.unknown["outofcode"] += [v]

            for v in lmodule:
                module = HDL(None)
                v = v.replace("\n", " ").split(";")
                for _ in v:
                    _ = _.strip().rstrip().replace(")", " ) ").replace("(", " ( ")
                    if module.name is None:
                        module.name = self.getNameWith(_, list(self.startsends["module"])[0])
                    if "parameter " in _:
                        x = {i.split("=")[0].replace("parameter", ""): {"val": i.split("=")[-1].strip().rstrip()} \
                             for i in re.split(",|\)|\(|#", _) if "parameter " in i}
                        z = {}
                        for xk, xv in x.items():
                            xx = re.findall("\w+", xk)
                            xv.update({"type": (xx[0] if xx.__len__() == 2 else "")})
                            z.update({xx[-1]: xv})
                        module.param.update(z)
                    if any([f"{i} " in _ for i in self.startsends["port"]]):
                        ptype = "wire"
                        pdir = ""
                        for m in _.split(","):
                            m = m.strip().replace("[", " [ ").replace("]", " ] ")
                            pdir_ = [i for i in self.startsends["port"] if i+" " in m]
                            if not (pdir or pdir_):
                                continue
                            m = m[m.index(pdir_[0]):] if pdir_ != [] else m
                            pdir = pdir_[0] if pdir_ != [] else pdir
                            ptype_ = [i for i in self.startsends["definition"] if i in m]
                            ptype = ptype if ptype_ == [] else ptype_[0] if pdir_ != [] else ptype
                            pname, x = self.cutout_code(m.replace(pdir, "").replace(ptype, ""), {"[": "]"})
                            pname = pname.strip().rstrip(")").rstrip()
                            if "[" in m and pdir_ != []:
                                pdim = x
                                array = [i for i in pdim if m.index(pname) > m.index(i)]
                                dimm = [i for i in pdim if i not in array]
                            elif "[" not in m and pdir_ != []:
                                array = []
                                dimm = []
                            port = PORT(pname)
                            port.type = ptype
                            port.dir = pdir
                            port.array = array
                            port.dimm = dimm
                            port.vlnv = "coffee2night::unknown::adhoc::1.0"
                            module.port.update({pname: port})
                file.inst += [module]
        return self.hdlfile


def dumppickle(sts):
    import pickle
    with open('hdl.class.pickle', 'wb') as fpick:
        pickle.dump(sts, fpick)


def loadpickle(sts=None):
    import pickle
    with open('hdl.class.pickle' if sts is None else sts, 'rb') as fpick:
        return pickle.load(fpick)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-s', '--src', help="source file", default="*.v")
    parser.add_argument('-D', '--define', help="source file", default="")
    parser.add_argument('-o', '--outdir', action="store", help="directory for result", default="OUT_VPARSE")
    args = parser.parse_args()

    class_hdlfile = verilog_parser(args)
    dumppickle(class_hdlfile)
    # mhdlfile_ = loadpickle()
    a = 0
