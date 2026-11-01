"""User-supplied September 2026 suit sources. Originals are copied byte-for-byte."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'References'/'Eastern_20260920'
SOURCES=[
 ('female_yellow_blue_sheet','130e22db-9347-49f6-9c45-55a17a759752'),
 ('female_red_tonal_kameez_front','0d13bbc5-5c7b-4934-9919-8147ce47e43e'),
 ('female_red_tonal_kameez_back','a11473d3-22a9-4d72-8270-51a46be6cb56'),
 ('female_red_tonal_trouser_front','928987ab-dfbd-4fef-92b6-474666a38ef8'),
 ('female_red_tonal_trouser_back','c85c9fa3-0cce-40b7-a214-e5a5bef6b7e3'),
 ('female_red_gold_kameez_front','05dc3190-98e8-4e9f-978d-77aa31204038'),
 ('female_red_gold_kameez_back','235bd92b-c556-481c-9080-e64a3a89f9d1'),
 ('female_red_gold_trouser_front','078782df-1fe5-4525-9cef-75338e4b894e'),
 ('female_red_gold_trouser_back','30f0ccfe-f0db-40d7-b915-fa0d1014b91a'),
 ('female_green_gold_kameez_front','1a1e9615-180d-4166-a969-b0e91fcf09a9'),
 ('female_green_gold_kameez_back','4710c678-0373-4bf1-934d-653d4afaf4c6'),
 ('female_green_gold_shalwar_front','184a5e82-917b-4fab-92bd-26215e0551f9'),
 ('female_green_gold_shalwar_back','f385931d-2d0a-4e91-b3d8-8a6829a2ec81'),
 ('male_blue_kurta_sheet','1831ec97-7906-4edb-951c-0294419e991f'),
 ('male_blue_shalwar_sheet','842e9fb7-aec2-46a6-b63c-c6a09b32762a'),
 ('male_black_kurta_sheet','c642a007-3bce-43cd-8aa3-9906b626f990'),
 ('male_black_shalwar_sheet','ec54d69f-95cf-4421-adba-30296cb3de96'),
 ('male_green_kurta_sheet','ffb150ba-9bd0-4af7-96f3-23a58e3d8ef6'),
 ('male_green_shalwar_sheet','1aeeddcd-d80c-4bbb-8b3b-bf558d7664a2')]

def archive():
    DEST.mkdir(parents=True,exist_ok=True);rows=[]
    for name,identifier in SOURCES:
        src=Path('C:/Users/eisha/AppData/Local/Temp')/('codex-clipboard-'+identifier+'.png');dst=DEST/(name+'.png')
        if not dst.exists():shutil.copy2(src,dst)
        digest=hashlib.sha256(dst.read_bytes()).hexdigest()
        if src.exists() and digest!=hashlib.sha256(src.read_bytes()).hexdigest():raise ValueError('Archived source mismatch: '+name)
        rows.append({'name':name,'file':dst.name,'sha256':digest,'original':str(src)})
    (DEST/'sources.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');return rows

def specifications():
    specs=[]
    def part(name,category,front,back=None,regions=None):
        return {'name':name,'category':category,'front':str(DEST/(front+'.png')),
                'back':str(DEST/((back or front)+'.png')),'regions':regions}
    for label,prefix,lower in [('Red tonal','female_red_tonal','trouser'),('Red gold','female_red_gold','trouser'),('Green gold','female_green_gold','shalwar')]:
        specs.append({'key':prefix,'name':label,'profile':'FEMALE','parts':[
            part(label+' Kameez','FEMALE_KAMEEZ',prefix+'_kameez_front',prefix+'_kameez_back'),
            part(label+' '+lower.title(),'FEMALE_'+lower.upper(),prefix+'_'+lower+'_front',prefix+'_'+lower+'_back')]})
    for label,columns in [('Yellow',(0,1)),('Blue floral',(2,3))]:
        upper=[];lower=[]
        for c in columns:
            upper.append(((7+368*c)/1477,42/1065,(369+368*c)/1477,506/1065))
            lower.append(((7+368*c)/1477,590/1065,(369+368*c)/1477,999/1065))
        specs.append({'key':'female_'+label.lower().replace(' ','_'),'name':label,'profile':'FEMALE','parts':[
            part(label+' Kameez','FEMALE_KAMEEZ','female_yellow_blue_sheet',regions=upper),
            part(label+' Shalwar','FEMALE_SHALWAR','female_yellow_blue_sheet',regions=lower)]})
    for color in ('blue','black','green'):
        regions=[(.035,.062,.495,.969),(.505,.062,.98,.969)]
        specs.append({'key':'male_'+color,'name':color.title()+' suit','profile':'MALE','parts':[
            part(color.title()+' Kurta','KURTA','male_'+color+'_kurta_sheet',regions=regions),
            part(color.title()+' Shalwar','SHALWAR','male_'+color+'_shalwar_sheet',regions=regions)]})
    return specs
