#!/usr/bin/env python3

__author__ = "dyxn.tonightcoffee@gmail.com"
__authors__ = ["dyxn.tonightcoffee@gmail.com"]
__contact__ = "dyxn.tonightcoffee@gmail.com"
__copyright__ = "Copyright 2023, free"
__credits__ = ["dyxn.tonightcoffee@gmail.com"]
__date__ = "2023/05/06"
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


def getinstsfiles(src, insts):
    command = f"find {src}/. | xargs egrep -s '{insts.replace('*', '').replace(',', '|')}'"
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output
    result = result.decode("utf-8")

    vfiles = {}
    for i in result.split("\n"):
        f = i.split(":")[0]
        if (f.endswith(".v") or f.endswith(".sv")) and os.path.isfile(f):
            if "module " not in i:
                if f not in vfiles:
                    vfiles[f] = []
                vfiles[f] += [i[f.__len__()+1:].strip().rstrip()]
    return vfiles


def getinstconn(hdl, dinst):
    instconn = {}
    for f in hdl.hdlfile:
        instconn[f.name] = {}
        for insts in f.inst:
            for ik, iv in insts.inst.items():
                for i in dinst.split(","):
                    if "*" in i:
                        pattern = i.replace("*", "([\w\d_]+)")
                        if re.findall(rf"{pattern}", iv["module"]):
                            instconn[f.name][ik] = iv
                    else:
                        if iv["module"] == i:
                            instconn[f.name][ik] = iv
    return instconn


def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-s', '--src', help="source file", default="*.v")
    parser.add_argument('-i', '--inst', help="module name to find", default="")
    parser.add_argument('-f', '--fileformat', help="pickle/json", default="json")
    parser.add_argument('-D', '--define', help="source file", default="")
    parser.add_argument('-o', '--outdir', action="store", help="directory for result", default="OUT_INSTCONN")
    parser.add_argument('-a', '--argument', help="more argument. conn/", default="")
    args = parser.parse_args()
    return args


def main(args):
    instfiles = getinstsfiles(args.src, args.inst)
    vargs = verilog_parser.Namespace()
    vargs.src = "\n".join(list(instfiles))
    hdl = verilog_parser.verilog_parser(vargs)
    instconn = getinstconn(hdl, args.inst)

    if args.fileformat:
        os.makedirs(args.outdir, exist_ok=True)
    if "json" in args.fileformat:
        import json
        with open(f"{args.outdir}/instconn_{args.inst.replace('*', '_')}.json", "w", encoding="utf8") as fp:
            json.dump(instconn, fp, indent=4)
            print(json.dumps(instconn))
    return instconn


if __name__=="__main__":
    main(getArgs())
