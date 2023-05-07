#!/usr/bin/env python3

__author__ = "dyxn.tonightcoffee@gmail.com"
__authors__ = ["dyxn.tonightcoffee@gmail.com"]
__contact__ = "dyxn.tonightcoffee@gmail.com"
__copyright__ = "Copyright 2023, free"
__credits__ = ["dyxn.tonightcoffee@gmail.com"]
__date__ = "2023/05/07"
__deprecated__ = False
__email__ = "dyxn.tonightcoffee@gmail.com"
__license__ = "MIT"
__maintainer__ = "developer"
__status__ = "Testing"
__version__ = "0.0.1"


import re
import os
import sys
import copy
import getinstconn
thisdir = "/".join(getinstconn.__file__.split("/")[:-1])
sys.path.append(f"{thisdir}/..")
from verilog_parser import verilog_parser
import subprocess


def getfilelist(src, insts):
    reginst = insts.replace('*', '[\w\d_]+').replace(' ', '\s*')
    command = f"find {src}/. | xargs grep -s -P '{reginst}'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    results = result.stdout

    vfiles = []
    for i in results.split("\n"):
        f = i.split(":")[0]
        if (f.endswith(".v") or f.endswith(".sv")) and os.path.isfile(f):
            vfiles += [f]
    return vfiles


def addcode(fl, jc):
    lsigs, lsigBase, lsigRegex, lcodecommend = [], [], [], []
    for _ in [jc["codecommand"], jc["seqcodecommand"]]:
        sigs = re.findall(r"(\*[\w\d_]+\*)", _)
        assert all(["*" not in i[1:-1] for i in sigs]), "ERR:: Can support only set both prefix and postfix to '*'"
        sigBase = [i.strip("*").rstrip("*") for i in sigs]
        sigRegex = [i.replace("*", "([\w\d_]+)?") for i in sigs]

        lsigs += [sigs]
        lsigBase += [sigBase]
        lsigRegex += [sigRegex]
        lcodecommend += [_]

    from verilog_parser import verilog_parser
    # dct = verilog_parser.Namespace()
    dct = {}
    for f in fl:
        vargs = verilog_parser.Namespace()
        vargs.src = "\n".join(list(fl))
        hdl = verilog_parser.verilog_parser(vargs)
        for fh in hdl.hdlfile:
            assert os.path.isfile(fh.name), f"ERR::{fh.name} is not a file."
            with open(fh.name, "r+", encoding="utf8") as rtls:
                rtl = rtls.read().replace("\t", "    ")
            dct[fh.name] = {"fixsGroup": [], "rtl": ""}
            for ninst, inst in enumerate(fh.inst):
                for n, sigs in enumerate(lsigs):
                    sigBase, sigRegex, codecommend = lsigBase[n], lsigRegex[n], lcodecommend[n]
                    code2add = []
                    lports = [i for i in inst.port]
                    lport = [i for i in lports if any([re.findall(rf"{j}", i) for j in sigRegex])]
                    dct[fh.name]["fixsGroup"] = {":".join(re.split(rf"{'|'.join(sigBase)}", i)):
                                         re.split(rf"{'|'.join(sigBase)}", i) for i in lport}
                    for fk, fv in dct[fh.name]["fixsGroup"].items():
                        fsg = fv
                        if any([f"{fsg[0]}{i}{fsg[1]}" not in lports for i in sigBase]):
                            print(f"INFO:: {fsg} pattern have not enough signal for adding code.")
                            continue
                        code2add += [codecommend.replace("%ENUM_CNT%", str(ninst + 1))]
                        for n, sig in enumerate(sigs):
                            code2add[-1] = code2add[-1].replace(sig, f"{fsg[0]}{sigBase[n]}{fsg[1]}")

                    code_adding = jc["comments"].format("\n".join(code2add)) + "\nendmodule"
                    rtl = rtl.replace("endmodule", code_adding)
            dct[fh.name]["rtl"] = rtl
    return dct


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    import json
    jcommand = json.load(open(args.jsoncommand, "r", encoding="utf8"))
    flielist = getfilelist(args.src, jcommand["module"])
    dct_results = addcode(flielist, jcommand)
    os.makedirs(f"{args.outdir}", exist_ok=True)
    for fk, fv in dct_results.items():
        with open(f"{args.outdir}/{fk.split('/')[-1]}", "w", encoding="utf8") as fp:
            fp.write(fv["rtl"])
    return


def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-s', '--src', help="source folder", default=".")
    parser.add_argument('-j', '--jsoncommand', help="", default="")
    parser.add_argument('-o', '--outdir', action="store", help="directory for result", default="OUT_VCODE")
    parser.add_argument('-a', '--argument', help="more argument. conn/", default="")
    args = parser.parse_args()
    return args


if __name__=="__main__":
    main(getArgs())
