#!/usr/bin/python3

# Copyright (c) 2023 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

from mako.template import Template

from collections import OrderedDict
from collections import namedtuple

import argparse
import re
import sys
import urllib
import xml.etree.ElementTree as etree

# parse_xml - Helper function to parse the XML file from a URL or local file.
def parse_xml(path):
    file = urllib.request.urlopen(path) if path.startswith("http") else open(path, 'r')
    with file:
        tree = etree.parse(file)
        return tree

# noneStr - returns string argument, or "" if argument is None.
def noneStr(s):
    if s:
        return s
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-registry', action='store',
                        default='cl.xml',
                        help='Use specified registry file instead of cl.xml')
    parser.add_argument('-o', action='store', dest='directory',
                        default='.',
                        help='Create target and related files in specified directory')

    args = parser.parse_args()

    specpath = args.registry
    #specpath = "https://raw.githubusercontent.com/KhronosGroup/OpenCL-Docs/master/xml/cl.xml"

    print('Parsing XML file from: ' + specpath)
    spec = parse_xml(specpath)

    # Generate the API function signatures dictionary:
    apisigs = OrderedDict()
    ApiSignature = namedtuple('ApiSignature', 'Name RetType Params')
    ApiParam = namedtuple('ApiParam', 'Type TypeEnd Name')
    print('Generating API signatures dictionary...')
    for command in spec.findall('commands/command'):
        proto = command.find('proto')
        ret = noneStr(proto.text)
        name = ""
        params = ""
        for elem in proto:
            if elem.tag == 'name':
                name = noneStr(elem.text) + noneStr(elem.tail)
            else:
                ret = ret + noneStr(elem.text) + noneStr(elem.tail)
        ret = ret.strip()
        name = name.strip()

        plist = []
        for param in command.findall('param'):
            ptype = noneStr(param.text)
            ptypeend = ""
            pname = ""
            for elem in param:
                if elem.tag == 'name':
                    pname = noneStr(elem.text)
                    ptypeend = noneStr(elem.tail)
                else:
                    ptype = ptype + noneStr(elem.text) + noneStr(elem.tail)
            ptype = ptype.strip()
            ptypeend = ptypeend.strip()
            pname = pname.strip()
            plist.append(ApiParam(ptype, ptypeend, pname))
        apisigs[name] = ApiSignature(name, ret, plist)

    # Generate the API versions dictionary:
    apivers = OrderedDict()
    print('Generating API versions dictionary...')
    for feature in spec.findall('feature'):
        version = noneStr(feature.get('name'))
        for function in feature.findall('require/command'):
            name = function.get('name')
            apivers[name] = version

    # Create the dispatch header from the API dictionary:
    print('Generating dispatch.h...')
    codegen_template = Template(filename='dispatch.h.mako', input_encoding='utf-8')
    text = codegen_template.render(
        spec=spec,
        apisigs=apisigs,
        apivers=apivers)
    text = re.sub(r'\r\n', r'\n', text)
    with open(args.directory + '/dispatch.h', 'w') as gen:
        gen.write(text)
