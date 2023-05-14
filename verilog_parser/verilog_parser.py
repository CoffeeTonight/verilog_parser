#!/usr/bin/env python3

__author__ = "dyxn.tonightcoffee@gmail.com"
__authors__ = ["dyxn.tonightcoffee@gmail.com"]
__contact__ = "dyxn.tonightcoffee@gmail.com"
__copyright__ = "Copyright 2023, free"
__credits__ = ["dyxn.tonightcoffee@gmail.com"]
__date__ = "2023/04/30"
__deprecated__ = False
__email__ = "dyxn.tonightcoffee@gmail.com"
__license__ = "MIT"
__maintainer__ = "developer"
__status__ = "Testing"
__version__ = "0.0.1"


import re
import os
import copy
import threading


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class HDL:
    def __repr__(self):
        return f'{self.name}'

    def __init__(self, name):
        self.name = name
        for i in ["path", "codemodule", "param", "inst", "port", "comment", "preprocessor", "subroutine", "localparam",
                  "wire", "reg", "assign", "always", "definition", "metaIF", "inlinedefine", "unknown", "endmodule",
                  "posedge", "negedge", "clock", "reset"]:
            self.__setattr__(i, {})

class IPXACT:
    def __repr__(self):
        return f'{self.vlnv}'

    def __init__(self):
        for i in ["vlnv", "logicalName", "prefix", "postfix", "metaname"]:
            self.__setattr__(i, str)


class IPXACTlib:
    def __init__(self):
        self.busDef = {}
        self.absDef = {}
        self.absDef2port = {"general::system::clock_rtl::1.0": ["clock"],
                            "general::system::reset_rtl::1.0": ["reset", "soft_reset", "sw_reset"],
                            "general::system::intr_rtl::1.0": ["intr", "irq", "fiq"],
                            }
    def port2vlnv(self, port):
        for k, v in self.absDef2port.items():
            for p in v:
                if port == p:
                    return k
        return None


class PORT:
    def __repr__(self):
        return f'{self.ipxact}'

    def __init__(self, name):
        self.name = name
        self.ipxact = IPXACT()
        for i in ["dir", "dimm", "array", "type"]:
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
        self.ipxactlib = IPXACTlib()
        self.define = {}
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
                f = f.readlines()
                self.filelist = [f"{os.path.abspath(i.strip().rstrip())}"
                                 for i in f if os.path.isfile(i.strip().rstrip())]

    def updateDefine(self):
        if hasattr(self.args, "define"):
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
                try:
                    idx_end = code[k:].index(conts[v])
                    cutout_code += [code[k:k + idx_end + conts[v].__len__()]]
                    code = code[:k] + code[k + idx_end + conts[v].__len__():]
                    break
                except:
                    cutout_code += [code[k:]]
                    code = code[:k]
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
        pattern_module = r"module\s+([\w\d_]+)\s*(#\s*\(\s*parameter\s+.*\))?\s*\((.*)\)"
        pattern_mparam = r"\bparameter\b\s.*"
        pattern_param = r"parameter\s+(\w+\s*)?([\w\d_]+)\s*=\s*(.*)"
        pattern_lparam = r"localparam\s+(\w+\s*)?([\w\d_]+)\s*=\s*(.*)"
        pattern_portb = r"(input|output|inout|buffer)\s+(wire\s|reg\s)?\s*(\[.*\])?\s*([\d\w_]+)(\[.*\])?"
        pattern_conn = r"\.(?P<name>\w+)\s*\((?P<value>.*?)\)"
        pattern_inst = r"([\w\d_]+)(\s*#\s*\(\s*\..*\))?\s*([\w\d_]+)\s*\(\s*\.(.*)\)"
        pattern_endmodule = r"\bendmodule\b"
        pattern_always = r"always\s*@\s*\((?P<sensitivelist>.*)\)"
        pattern_begin = r"\bbegin\b"
        pattern_end = r"\bend\b"
        hdlfile = []
        for file in copy.deepcopy(self.hdlfile):
            hdlfile += [self.extractFlist(file)]
        for file in hdlfile:
            hdl = open(file.name, "r", encoding="utf8").read().replace("\t", "    ")

            #todo: find module
            ncode, file.comment = self.cutout_code(hdl, self.startsends["comment"])

            # todo: handles ifdef
            ncode, file.definition = self.conditionGen(ncode, self.startsends["ifdef"], self.define)
            defcode, code_seqs = self.merge_nestedcode(ncode, file.definition)
            assert ncode.__len__() == defcode.__len__(), "ERR::Check Tool"

            outofcode, lmodule = self.cutout_code(defcode.replace("\t", " ").replace("\\\n", " "),
                                                  self.startsends["module"])

            for v in outofcode.replace("\t", "").split("\n"):
                v = v.strip().rstrip()
                if v == "":
                    continue
                #grep preprocessor
                if any([v.startswith(i) for i in list(self.startsends["preprocess"])]):
                    if "command" not in file.preprocessor:
                        file.preprocessor.update({"command": self.define, "code": []})
                    file.preprocessor["code"] += [v]
                else:
                    if "outofcode" not in file.unknown:
                        file.unknown.update({"outofcode": []})
                    file.unknown["outofcode"] += [v]

            begin_end_bal = 0
            on_always = False
            for v in lmodule:
                module = HDL(None)
                module.always = []
                module.localparam = []
                module.param = []
                module.wire = []
                module.reg = []
                module.assign = []
                module.__setattr__("genvar", [])
                module.__setattr__("generate", [])
                module.__setattr__("conditional", [])
                module.unknown = []
                module.path = file.name
                on_always = False
                exist_if = False
                vv = v.replace("\t", "    ").split(";")
                for n, __ in enumerate(vv):
                    for _ in re.sub(r"\bend\b", "$end;", __).split(";"):
                        _ = _.replace("\t", "    ").strip()
                        _t = _.replace("\n", " ").strip().rstrip()
                        if exist_if and _t.startswith("else "):
                            on_always = True
                        if _t.startswith("always"):
                            always = re.findall(pattern_always, _t, re.MULTILINE)
                        else:
                            always = None
                        if always or on_always:
                            if always:
                                exist_if = re.findall(r"\bif\b", _t)
                                _posedge = re.findall(r"(\bposedge\b)\s*([\w\d_]+)", _t, re.MULTILINE)
                                _negedge = re.findall(r"(\bnegedge\b)\s*([\w\d_]+)", _t, re.MULTILINE)
                                for _ps in _posedge + _negedge:
                                    sysport = "clock" if any([i in _ps[1].lower() for i in ["clk", "ck", "clock"]]) else "reset"
                                    _a = _ps[1].strip().rstrip() in list(module.port)
                                    if _a:
                                        module.port[_ps[1]].ipxact.vlnv = self.getVLNVwPort(sysport)
                                    if "posedge" == _ps[0]:
                                        if _ps[1] not in module.posedge:
                                            module.posedge[_ps[1]] = {"count": 0, "type": f"port_{sysport}" \
                                                                                    if _a else f"gated_{sysport}"}
                                        module.posedge[_ps[1]]["count"] += 1
                                        module.clock.update({_ps[1]: {} if _ps[1] not in module.clock else {}})
                                    else:
                                        if _ps[1] not in module.negedge:
                                            module.negedge[_ps[1]] = {"count": 0, "type": f"port_{sysport}" \
                                                                                    if _a else f"gated_{sysport}"}
                                        module.negedge[_ps[1]]["count"] += 1
                                        module.reset.update({_ps[1]: {} if _ps[1] not in module.reset else {}})
                            begin_val = re.findall(pattern_begin, _)
                            end_val = re.findall(pattern_end, _)
                            begin_end_bal += begin_val.__len__() - end_val.__len__()
                            if on_always:
                                module.always[-1] += _.replace("\t", "\n").replace("$end;", "end") + ";"
                            else:
                                if module.always.__len__() > 0:
                                    module.always[-1] = module.always[-1].replace("$end;", "end")
                                module.always += [_.replace("\t", "\n") + ";"]
                            on_always = begin_end_bal != 0
                            if not on_always:
                                module.always[-1] = module.always[-1].replace("$end;", "end")
                            continue
                        if not module.name and _.startswith("module "):
                            module_casts = re.findall(pattern_module, _t.replace("\n", " "), re.MULTILINE)
                        if module_casts:
                            module_cast = module_casts[0]
                            module.name = module_cast[0]
                            module.param = [re.findall(pattern_mparam, i, re.MULTILINE)[0]
                                            for i in module_cast[1].split(",")]
                            module.port = {}
                            try:
                                for p in module_cast[2].split(","):
                                    tpport = re.findall(pattern_portb, p, re.MULTILINE)[0]
                                    port = PORT(tpport[3])
                                    port.type = "wire " if tpport[1] == "" else tpport[1]
                                    port.dir = tpport[0]
                                    port.array = tpport[2]
                                    port.dimm = tpport[4]
                                    port.ipxact.vlnv = "all::unknown::adhoc::0.1"
                                    module.port[tpport[3]] = port
                            except:
                                module.port = {}
                            module.codemodule = _.replace("\t", "\n") + ";"
                            module_casts = None
                            continue
                        if [_t.startswith(i) for i in ["output ", "input ", "inout ", "buffer "]]:
                            ports = re.findall(pattern_portb, _t, re.MULTILINE)
                            if ports:
                                tpport = ports[0]
                                for p in tpport[3].split(","):
                                    port = PORT(p)
                                    port.type = "wire " if tpport[1] == "" else tpport[1]
                                    port.dir = tpport[0]
                                    port.array = tpport[2]
                                    port.dimm = tpport[4]
                                    port.ipxact.vlnv = "coffee2night::unknown::adhoc::1.0"
                                    module.port[p] = port
                                continue
                        escape = False
                        for i in ("genvar", "assign", "wire", "reg", "parameter", "localparam"):
                            if re.findall(rf"\b{i}\b", _t):
                                exec(f"module.{i} += [_]")
                                escape = True
                                break
                        if escape:
                            escape = False
                            continue
                        if "endmodule" in _t:
                            endmodule = re.findall(pattern_endmodule, _t)
                            if endmodule:
                                module.endmodule = {n: 'replace("\n", "\t").split(";")'}
                                continue
                        insts = re.findall(pattern_inst, _t, re.MULTILINE)
                        if insts:
                            inst = insts[0]
                            inst_module = inst[0]
                            params = re.findall(pattern_conn, inst[1], re.MULTILINE) if inst[1] else []
                            instance = inst[2]
                            instance_conn = re.findall(pattern_conn, inst[3], re.MULTILINE)
                            module.inst[instance] = {"code": _[_.index(inst_module):],
                                                     "module": inst_module, "module_param": params,
                                                     "inst": instance, "inst_conn": instance_conn}
                            continue
                        module.unknown += [_]
                file.inst += [module]
        self.hdlfile = hdlfile
        return self.hdlfile


    def getVLNVwPort(self, port):
        return self.ipxactlib.port2vlnv(port)


def dumppickle(args, sts='class.hdl.pickle'):
    import pickle
    if isinstance(sts, verilog_parser):
        sts = sts.args.src.split("/")[-1] + ".pickle"
    os.makedirs(args.outdir, exist_ok=True)
    with open(sts, 'wb') as fpick:
        pickle.dump(sts, fpick)
    print(f"\nINFO::Pickle file for Meta-data had stored, {sts}")


def loadpickle(sts='class.hdl.pickle'):
    import pickle
    if isinstance(sts, verilog_parser):
        sts = sts.args.src.split("/")[-1] + ".pickle"
    with open(sts, 'rb') as fpick:
        print(f"\nINFO::Pickle file for Meta data Loaded, {sts}")
        return pickle.load(fpick)


def main(args):
    class_hdlfile = verilog_parser(args)
    if args.pickle:
        dumppickle(args, class_hdlfile)
    return class_hdlfile


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-s', '--src', help="source file", default="*.v")
    parser.add_argument('-p', '--pickle', help="store meta as a pickle file", default="")
    parser.add_argument('-f', '--findcondition', help="Condition to find specific v file", default="")
    parser.add_argument('-D', '--define', help="source file", default="")
    parser.add_argument('-o', '--outdir', action="store", help="directory for result", default="OUT_VPARSE")
    parser.add_argument('-a', '--argument', help="more argument. profile/", default="")
    args = parser.parse_args()

    if "profile" in args.argument:
        import cProfile
        cProfile.run('[main(args) for i in range (0, 1024*10)]')
    else:
        main(args)
