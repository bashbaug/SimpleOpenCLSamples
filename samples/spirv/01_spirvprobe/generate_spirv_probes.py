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
    parser = argparse.ArgumentParser(description='Generate SPIR-V extension and capability probes')

    parser.add_argument('--grammar', metavar='<path>',
                        type=str, required=True,
                        help='input JSON grammar file')
    parser.add_argument('--output', metavar='<path>',
                        type=str, required=False,
                        help='output file path (default: stdout)')
    args = parser.parse_args()

    extensions = {}
    with open(args.grammar) as json_file:
        grammar_json = json.loads(json_file.read())
        for operand_kind in grammar_json['operand_kinds']:
            if not 'enumerants' in operand_kind:
                continue
            for enum in operand_kind['enumerants']:
                if not 'extensions' in enum:
                    continue
                for extension in enum['extensions']:
                    extensions.setdefault(extension, [])
                    if operand_kind['kind'] == 'Capability':
                        name = enum['enumerant']
                        extensions[extension].append(name)

    output = []
    output.append(header_text)
    for extension in sorted(extensions.keys()):
        output.append('')
        output.append('// {}:'.format(extension))
        output.append('SPIRV_PROBE_EXTENSION( {} )'.format(extension))
        for cap in sorted(extensions[extension]):
            output.append('SPIRV_PROBE_EXTENSION_CAPABILITY( {}, {} )'.format(extension, cap))

    if args.output:
        with open(args.output, 'w') as output_file:
            output_file.write('\n'.join(output))
    else:
        print('\n'.join(output))

if __name__ == '__main__':
    main()
