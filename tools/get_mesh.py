# -*- coding: utf-8 -*-
#
# get_mesh.py - extract and multilingualize MeSH terms
#
# Written in 2019 by Anonymous Author
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
#
# To the extent possible under law, the author(s) have dedicated all copyright
# and related and neighboring rights to this software to the public domain
# worldwide. This software is distributed without any warranty. You should have
# received a copy of the CC0 Public Domain Dedication along with this software.
# If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
#

import argparse
import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import json
import sys
from tqdm import tqdm
from lxml.html import fromstring

auth_uri = "https://utslogin.nlm.nih.gov"
#option 1 - username/pw authentication at /cas/v1/tickets
#auth_endpoint = "/cas/v1/tickets/"
#option 2 - api key authentication at /cas/v1/api-key
auth_endpoint = "/cas/v1/api-key"

content_uri = "https://uts-ws.nlm.nih.gov"
version = "current"
content_endpoint = "/rest/search/" + version
atom_endpoint = "/rest/content/" + version + "/CUI/"


class Authentication:
    #def __init__(self, username,password):
    def __init__(self, apikey):
        #self.username=username
        #self.password=password
        self.apikey = apikey
        self.service = "http://umlsks.nlm.nih.gov"

    def gettgt(self):
        #params = {'username': self.username,'password': self.password}
        params = {'apikey': self.apikey}
        h = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "python"
        }
        r = requests.post(auth_uri + auth_endpoint, data=params, headers=h)
        response = fromstring(r.text)
        ## extract the entire URL needed from the HTML form (action attribute) returned - looks similar to https://utslogin.nlm.nih.gov/cas/v1/tickets/TGT-36471-aYqNLN2rFIJPXKzxwdTNC5ZT7z3B3cTAKfSc5ndHQcUxeaDOLN-cas
        ## we make a POST call to this URL in the getst method
        tgt = response.xpath('//form/@action')[0]
        return tgt

    def getst(self, tgt):
        params = {'service': self.service}
        h = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-Agent": "python"
        }
        r = requests.post(tgt, data=params, headers=h)
        st = r.text
        return st


def add_args(parser):
    ## Required parameters
    parser.add_argument("--mesh",
                        default=None,
                        type=str,
                        required=True,
                        help="MESH XML file location")
    parser.add_argument("--apikey",
                        default=None,
                        type=str,
                        required=True,
                        help="UMLS API key")
    parser.add_argument("-t",
                        "--target-langs",
                        default=None,
                        required=True,
                        metavar="LANG",
                        nargs='+',
                        help="list of target languages")
    return parser


def umls_query(string, lang_list, AuthClient, tgt):
    result_dict = defaultdict(str)
    ticket = AuthClient.getst(tgt)

    query = {'string': string, 'ticket': ticket, 'pageNumber': 1}
    #query['includeObsolete'] = 'true'
    #query['includeSuppressible'] = 'true'
    #query['returnIdType'] = "sourceConcept"
    query['pageSize'] = 1
    query['sabs'] = "MSH"
    query['inputType'] = "atom"  #"sourceConcept"
    query['searchType'] = "exact"

    r = requests.get(content_uri + content_endpoint, params=query)
    r.encoding = 'utf-8'
    results = json.loads(r.text)["result"]["results"]
    ui = results[0]["ui"]

    ticket = AuthClient.getst(tgt)
    query = {'ticket': ticket, 'pageNumber': 1}
    atom_uri = content_uri + atom_endpoint + ui + "/atoms"

    r = requests.get(atom_uri, params=query)
    r.encoding = 'utf-8'
    results = json.loads(r.text)["result"]

    for result in results:
        if result["language"] in lang_list and result[
                "rootSource"][:3] == "MSH":
            result_dict[result["language"]] = result["name"]

    return result_dict


def mesh_heading(mesh_xml):
    mesh = []
    mesh2nums = defaultdict(list)
    root = ET.parse(mesh_xml).getroot()

    for entry in root.iter('DescriptorRecord'):
        mesh.append(entry[1][0].text)

        for tree_num_list in entry.iter('TreeNumberList'):
            for tree_num in tree_num_list.iter('TreeNumber'):
                mesh2nums[mesh[-1]].append(tree_num.text)

    # return mesh, mesh2tree_num
    return mesh, mesh2nums


def write_mesh_dict(mesh_list, mesh2nums, lang_list, out_file, AuthClient,
                    tgt):
    fps = {
        lg: open(out_file + '.' + lg, 'a', encoding="utf-8")
        for lg in lang_list + ['ENG', 'tree.num']
    }

    for i, string in enumerate(tqdm(mesh_list)):
        try:
            results = umls_query(string, lang_list, AuthClient, tgt)
            for lg in lang_list:
                fps[lg].write(results[lg] + '\n')
            fps['ENG'].write(string + '\n')
            fps['tree.num'].write(",".join(mesh2nums[string]) + '\n')
        except json.JSONDecodeError:
            print("\nJSONDecodeError occurred with #{:d}: {}".format(
                i, string),
                  file=sys.stderr)

    for fp in fps.values():
        fp.close()


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    AuthClient = Authentication(args.apikey)

    mesh, mesh2nums = mesh_heading(args.mesh)
    write_mesh_dict(mesh, mesh2nums, args.target_langs, args.mesh, AuthClient,
                    AuthClient.gettgt())


if __name__ == "__main__":
    main()
