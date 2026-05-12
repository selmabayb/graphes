#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mistral_llm import query_mistral

import networkx as nx
import random, math, time, multiprocessing as mp
import statistics, matplotlib.pyplot as plt
import hashlib, signal


# ================================================================
# SANDBOX
# ================================================================

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException()
signal.signal(signal.SIGALRM, timeout_handler)


# ================================================================
# BENCHMARK
# ================================================================

def generate_tree_graph(n): return nx.generators.trees.random_labeled_tree(n)
def generate_grid_graph(w,h): return nx.grid_2d_graph(w,h)
def generate_planar_graph(n,p=0.1):
    G = nx.random_geometric_graph(n,p)
    pos = nx.get_node_attributes(G,"pos")
    return nx.random_geometric_graph(n,p,pos=pos)
def generate_geometric_graph(n,r=0.15): return nx.random_geometric_graph(n,r)
def generate_powerlaw_graph(n): return nx.powerlaw_cluster_graph(n,2,0.2)
def generate_bipartite_graph(n1,n2,p): return nx.bipartite.random_graph(n1,n2,p)
def generate_adversarial_graph():
    G = nx.path_graph(50)
    for i in range(10):
        star = nx.star_graph(10)
        nx.relabel_nodes(star,lambda x,offset=i*20: x+offset,copy=False)
        G = nx.disjoint_union(G,star)
    return G

def build_benchmark():
    return [
        generate_tree_graph(150),
        generate_tree_graph(300),
        generate_grid_graph(20,20),
        generate_planar_graph(300),
        generate_geometric_graph(400),
        generate_powerlaw_graph(500),
        generate_bipartite_graph(200,200,0.05),
        generate_adversarial_graph()
    ]


# ================================================================
# EVALUATION
# ================================================================

def evaluate_objectives(G,S,t):
    if not isinstance(S,set): return (-1e9,-1e9,-1e9)
    conflict=sum(1 for u in S for v in G.neighbors(u) if v in S)//2
    return len(S), 1/(1+t), -conflict


def run_heuristic_llm(args):
    code,G=args
    local={}
    try:
        signal.alarm(2)
        start=time.time()
        exec(code,{"nx":nx,"random":random,"math":math},local)
        fun=local.get("mis_heuristic")
        if fun is None: return (-1e9,-1e9,-1e9),None
        S=fun(G)
        signal.alarm(0)
        return evaluate_objectives(G,S,time.time()-start),S
    except:
        return (-1e9,-1e9,-1e9),None


def evaluate_code(code,graphs):
    with mp.Pool(processes=8) as p:
        return p.map(run_heuristic_llm,[(code,G) for G in graphs])


# ================================================================
# MISTRAL – PARSING SÛR
# ================================================================

def extract_python_function(text):
    lines=text.splitlines()
    code=[]
    inside=False
    for l in lines:
        if l.strip().startswith("def mis_heuristic"):
            inside=True
        if inside:
            if l.strip().startswith("```"):
                break
            code.append(l)
    return "\n".join(code).strip()


def llm_prompt_template():
    return """
Tu es un expert en optimisation du MIS (Maximum Independent Set).

Ta tâche :
- produire une VARIATION ou une FUSION de stratégies
- utiliser propriétés de graphes (degrés, clustering, random…)
- MAXIMUM 10 lignes pour la fonction
- retourner un set()
- pas de print, pas d'I/O

Format STRICT :

def mis_heuristic(G):
    ...
    return S
"""


def prompts_to_code(pop):
    out=[]
    for p in pop:
        if p.strip().startswith("def mis_heuristic"):
            out.append(p)
        else:
            print("\n>>> Mistral génère une heuristique\n")
            raw=query_mistral(p)
            code=extract_python_function(raw)
            out.append(code)
    return out


# ================================================================
# EVOLUTION
# ================================================================

def hash_code(code): return hashlib.sha256(code.encode()).hexdigest()

def fitness(results):
    objs=[r[0] for r in results]
    return sum(o[0] for o in objs), statistics.mean(o[1] for o in objs), statistics.mean(o[2] for o in objs)


def evolve(pop,generations=5):
    graphs=build_benchmark()
    hall={}

    for gen in range(generations):
        print(f"\n=== GEN {gen} ===")
        scored=[]

        for code in pop:
            h=hash_code(code)
            if h not in hall:
                hall[h]={"code":code,"fitness":fitness(evaluate_code(code,graphs))}
            scored.append((hall[h]["fitness"],code))

        scored.sort(reverse=True,key=lambda x:x[0])
        print("Best:",scored[0][0])

        pop=[scored[i][1] for i in range(min(5,len(scored)))]
        while len(pop)<10: pop.append(pop[0])

    return max(hall.values(),key=lambda x:x["fitness"])


# ================================================================
# VISUALISATION
# ================================================================

def visualize(G,S,title="Final"):
    pos=nx.spring_layout(G)
    colors=["red" if n in S else "lightblue" for n in G.nodes()]
    plt.figure(figsize=(7,7))
    nx.draw(G,pos,node_color=colors,node_size=70)
    plt.title(title)
    plt.show()

def visualize_best(code):
    G=generate_geometric_graph(200)
    (_,_,_),S=run_heuristic_llm((code,G))
    visualize(G,S,"Best MIS")


# ================================================================
# MAIN
# ================================================================

if __name__=="__main__":
    print("\n--- AUTO-FUNSEARCH MIS v5 — MISTRAL MODE ---\n")

    population=[
        llm_prompt_template(),
        llm_prompt_template(),
        llm_prompt_template()
    ]

    population=prompts_to_code(population)
    result=evolve(population, generations=5)

    print("\n=== BEST HEURISTIC ===\n")
    print(result["code"])

    visualize_best(result["code"])
