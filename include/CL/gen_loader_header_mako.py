#!/usr/bin/python3

# Copyright (c) 2019 >>>TBD<<<
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mako.template import Template

from collections import OrderedDict
from collections import namedtuple

import argparse
import sys
import urllib
import xml.etree.ElementTree as etree
import urllib.request

# APIs to skip - they need to be done "manually":
apiskip = {
    'clGetPlatformIDs',                 # calls into the ICD loader
    'clGetExtensionFunctionAddress',    # could call into the ICD loader or return NULL
    'clUnloadCompiler',                 # unconditionally returns CL_SUCCESS
    }

# Handles, and mappings to errors when handles are invalid:
apihandles = {
    'cl_command_queue'  : 'CL_INVALID_COMMAND_QUEUE',
    'cl_context'        : 'CL_INVALID_CONTEXT',
    'cl_device_id'      : 'CL_INVALID_DEVICE',
    'cl_event'          : 'CL_INVALID_EVENT',
    'cl_kernel'         : 'CL_INVALID_KERNEL',
    'cl_mem'            : 'CL_INVALID_MEM_OBJECT',
    'cl_platform_id'    : 'CL_INVALID_PLATFORM',
    'cl_program'        : 'CL_INVALID_PROGRAM',
    'cl_sampler'        : 'CL_INVALID_SAMPLER',
    }

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
    #specpath = "https://raw.githubusercontent.com/KhronosGroup/OpenCL-Registry/master/xml/cl.xml"

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

    # Create the loader header from the API dictionary:
    test = open(args.directory + '/cl_static.h', 'wb')
    cl_static_h_template = Template(filename='cl_static.h.mako')
    test.write(
        cl_static_h_template.render_unicode(
            spec=spec,
            apiskip=apiskip,
            apihandles=apihandles,
            apisigs=apisigs,
            apivers=apivers).
        encode('utf-8', 'replace'))
