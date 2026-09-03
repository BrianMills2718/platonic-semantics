#!/usr/bin/env python3
"""
Optional external check of the curated relation probe set against ConceptNet.
This does NOT use ConceptNet to position points; it only records whether an
equivalent English edge is present in the external knowledge graph.
"""
from __future__ import annotations
import argparse,csv,json,time,urllib.parse,urllib.request
from pathlib import Path

def slug(s):
    return s.lower().replace(" ","_")

def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"semantic-relational-atlas-pilot/0.1"})
    with urllib.request.urlopen(req,timeout=20) as r:
        return json.load(r)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--concepts",default="benchmark/concepts.csv")
    ap.add_argument("--relations",default="benchmark/relations.csv")
    ap.add_argument("--out",default="outputs/conceptnet_validation.csv")
    ap.add_argument("--sleep",type=float,default=.15)
    args=ap.parse_args()
    concepts={r["concept_id"]:r for r in csv.DictReader(open(args.concepts,encoding="utf-8"))}
    rows=list(csv.DictReader(open(args.relations,encoding="utf-8")))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["source","target","relation","conceptnet_found","max_weight"])
        w.writeheader()
        for i,r in enumerate(rows,1):
            a=slug(concepts[r["source"]]["en"]);b=slug(concepts[r["target"]]["en"])
            rel=urllib.parse.quote("/r/"+r["relation"],safe="/")
            url=("https://api.conceptnet.io/query?start=/c/en/"+urllib.parse.quote(a)+
                 "&end=/c/en/"+urllib.parse.quote(b)+"&rel="+rel+"&limit=20")
            found=False;weight=0.0
            try:
                data=fetch(url)
                edges=data.get("edges",[])
                found=bool(edges)
                weight=max([float(e.get("weight",0)) for e in edges] or [0.0])
            except Exception as exc:
                print("warning",r["source"],r["target"],exc)
            w.writerow({"source":r["source"],"target":r["target"],"relation":r["relation"],
                        "conceptnet_found":int(found),"max_weight":weight})
            if i%10==0: print(i,"/",len(rows))
            time.sleep(args.sleep)

if __name__=="__main__":
    main()
