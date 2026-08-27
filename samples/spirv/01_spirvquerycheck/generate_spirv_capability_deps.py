#!/usr/bin/env python3

# Copyright (c) 2026 Ben Ashbaugh
#
# SPDX-License-Identifier: MIT

import argparse
import json

header_text = """\
// This file is generated from the SPIR-V JSON grammar file.
// Please do not edit it directly!
"""

def main():
    parser = argparse.ArgumentParser(description='Generate SPIR-V extension and version dependencies for SPIR-V capabilities')

    parser.add_argument('--grammar', metavar='<path>',
                        type=str, required=True,
                        help='input JSON grammar file')
    parser.add_argument('--output', metavar='<path>',
                        type=str, required=False,
                        help='output file path (default: stdout)')
    args = parser.parse_args()

    dependencies = {}
    capabilities = []
    with open(args.grammar) as json_file:
        grammar_json = json.loads(json_file.read())
        for operand_kind in grammar_json['operand_kinds']:
            if operand_kind['kind'] == 'Capability':
                for cap in operand_kind['enumerants']:
                    capname = cap['enumerant']
                    capabilities.append(capname)
                    dependencies[capname] = {}
                    dependencies[capname]['extensions'] = cap['extensions'] if 'extensions' in cap else []
                    dependencies[capname]['version'] = ("SPIR-V_" + cap['version']) if 'version' in cap and cap['version'] != 'None' else ""

    capabilities.sort()

    output = []
    output.append(header_text)
    for cap in capabilities:
        deps = dependencies[cap]
        if deps['version'] != "":
            output.append('SPIRV_CAPABILITY_VERSION_DEPENDENCY( {}, "{}" )'.format(cap, deps['version']))
        for ext in deps['extensions']:
            output.append('SPIRV_CAPABILITY_EXTENSION_DEPENDENCY( {}, "{}" )'.format(cap, ext))

    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write('\n'.join(output))
    else:
        print('\n'.join(output))

if __name__ == '__main__':
    main()
