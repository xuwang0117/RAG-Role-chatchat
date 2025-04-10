'''
Author: sdw yz1001sundaiwn@163.com
Date: 2024-11-11 19:11:00
LastEditors: shihongliang shihongliang@wyy.com
LastEditTime: 2024-12-05 15:42:16
FilePath: /Langchain-Chatchat/server/person_info_util.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import requests, json
import random
# from configs.person_desc import mysql_config
from server.utils import get_respond_httpx

def get_value_str(value):
    if isinstance(value, list):
        return ",".join(value)
    elif isinstance(value, str):
        return value
    else:
        raise TypeError(f"value类型不支持,需要是list或str, 当前类型为{type(value)}")


def get_org_info(id):
    try:
        resp = requests.post(url="http://present.a.wkycloud.com/present/api/auth/oauth2/token?grant_type=password",
                    headers={"User-Agent": "Apifox/1.0.0(https://apifox.com)",
                            "Authorization": "Basic Y29tcG9uZW50Om1hbmFnZW1lbnQ="},
                    data={"username":"systema2","password":"flPxo5o/eyf5GBi/UBBlzmHODlw=","scope":"server"})
        # print(resp.text)
        # print('access_token' in json.loads(resp.text))
        access_token = json.loads(resp.text)['access_token']
        # print(access_token)
        headers = {
            'Connection': 'keep-alive',
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {access_token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
            'Referer': 'http://present.a.wkycloud.com',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-CA;q=0.8,en;q=0.7'
        }


        org_attr = {}
        
        url = 'http://present.a.wkycloud.com/gene/api/situation/_loop_playback'
        response = requests.post(url, headers=headers, json={"id":id,"if_check":"true","label":"org"})
        abstract = response.json()["data"]["abstract"]
        
        org_attr['姓名'] = response.json()["data"]["name"]
        org_attr['摘要'] = abstract
        
        for att in response.json()['data']["detail"]:
            if att['cn'] == "意识形态":
                org_attr['意识形态'] = get_value_str(att['val'])
            if att['cn'] == '其他名称':
                org_attr['其他名称'] = get_value_str(att['val'])
            if att['cn'] == "简介":
                org_attr['简介'] = get_value_str(att['val'])
            if att['cn'] == "组织总部":
                org_attr['组织总部'] = get_value_str(att['val'])
            if att['cn'] == "创建时间 ":
                org_attr['创建时间'] = get_value_str(att['val'])
            if att['cn'] == "所属国家 ":
                org_attr['所属国家'] = get_value_str(att['val'])
    except:
        org_attr = {}
    return org_attr


async def get_user_info(id):
    try:
        person_attr_respond = await get_respond_httpx({"id": id}, f"{mysql_config['url']}/experts/info")
        person_attr = person_attr_respond['data']
        return person_attr
    except:
        person_attr = {}
        
    try:
        resp = requests.post(url="http://present.a.wkycloud.com/present/api/auth/oauth2/token?grant_type=password",
                    headers={"User-Agent": "Apifox/1.0.0(https://apifox.com)",
                            "Authorization": "Basic Y29tcG9uZW50Om1hbmFnZW1lbnQ="},
                    data={"username":"systema2","password":"flPxo5o/eyf5GBi/UBBlzmHODlw=","scope":"server"})
        # print(resp.text)
        # print('access_token' in json.loads(resp.text))
        access_token = json.loads(resp.text)['access_token']
        # print(access_token)
        headers = {
            'Connection': 'keep-alive',
            'Accept': 'application/json, text/plain, */*',
            'Authorization': f'Bearer {access_token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
            'Referer': 'http://present.a.wkycloud.com',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-CA;q=0.8,en;q=0.7'
        }


        person_attr = {}
        
        url = 'http://present.a.wkycloud.com/gene/api/situation/_loop_playback'
        response = requests.post(url, headers=headers, json={"id":id,"if_check":"true","label":"human"})
        abstract = response.json()["data"]["abstract"]
        
        person_attr['姓名'] = response.json()["data"]["name"]
        person_attr['摘要'] = abstract
        
        for att in response.json()['data']["detail"]:
            if att['cn'] == "所属政党":
                person_attr['所属政党'] = get_value_str(att['val'])
            if att['cn'] == '担任职务':
                person_attr['担任职务'] = get_value_str(att['val'])
            if att['cn'] == "简介":
                person_attr['简介'] = get_value_str(att['val'])
            if att['cn'] == "就职部门/公司":
                person_attr['单位'] = get_value_str(att['val'])
            if att['cn'] == "职业 ":
                person_attr['职称'] = get_value_str(att['val'])
        
        
        url = 'http://present.a.wkycloud.com/gene/api/situation/human_cognition'
        
        response = requests.post(url, headers=headers, json={"mid":id})
        
        # print("*"*8)
        # print(response.json())
        
        
        political_tendency = response.json()["data"]['political_tendency']
        cn_related_stand = response.json()["data"]['cn_related_stand']
        lang_styles = response.json()["data"]['lang_styles']
        # print("political_tendency: ", political_tendency)
        # print("cn_related_stand: ", cn_related_stand)
        # print("lang_styles: ", lang_styles)
        
        person_attr['政治倾向'] = political_tendency
        person_attr['相关立场'] = cn_related_stand
        person_attr['语言风格'] = lang_styles
    except:
        person_attr = {}
        
 
    person_attr = get_org_info(id)
        
    return person_attr
    


async def get_info_with_name_from_db(name:str):
    out = []
    try:
        respond = await get_respond_httpx({}, f"{mysql_config['url']}/experts/search_with_name?name={name}")
        if len(respond):
            out = respond['data']
    except:
        pass
    return out


async def get_expert_list_from_db(fields=[], en=False, recom=False, region='all'):
    person_list_info = []
    try:
        person_list_info_respond = await get_respond_httpx({"fields": fields, 'en':en, 'recom': recom, 'region':region}, f"{mysql_config['url']}/experts/expert_list")
        person_list_info = person_list_info_respond['data']
    except:
        pass
    return person_list_info


async def get_special_num_experts_from_db(num=20):
    expert_desc_part = []
    try:
        expert_desc_part_respond = await get_respond_httpx({"num":num}, f"{mysql_config['url']}/experts/special_num_experts")
        expert_desc_part = expert_desc_part_respond['data']
    except:
        pass
    return expert_desc_part
    
    
# expert_desc = {}

# for item in person_img['person']:
#     expert_info_single = get_user_info(id=item['id'])
#     expert_info_single.update(item)
#     expert_desc[item['name']] = expert_info_single

# def get_coarse_ranking():
#     """粗排阶段"""
#     item_list = random.sample(person_img['person'], 20)
#     coarse_rank_out = {item['name']: expert_desc[item['name']] for item in item_list}
#     return coarse_rank_out

if __name__ == "__main__":
    out = get_user_info("Q5090717")
    from pprint import pprint
    pprint(out)