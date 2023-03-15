#!/usr/bin/python

# from msilib.schema import File
from audioop import reverse
from cgi import print_environ
from copy import deepcopy
from distutils import errors
from genericpath import exists
from tokenize import Number
from tracemalloc import start
import typing
from collections import Counter

from curses import meta
from datetime import date, datetime
from decimal import DivisionByZero
from inspect import trace
from re import search
import math
import hashlib
from sklearn import preprocessing

# from nscl_algo import NSCLAlgo
# from nscl_algo import NSCLAlgo

# from nscl_algo import NSCLAlgo
import pprint
import os
from re import split, template
import time
import sys
import json
import pprint
import subprocess

# from subprocess import Popen
from typing import NewType

# from jinja2.defaults import NEWLINE_SEQUENCE

from networkx.algorithms.planarity import Interval
from networkx.generators.geometric import random_geometric_graph
from pandas.core.algorithms import take
from nscl import NSCL
import matplotlib.pyplot as plt
from nscl_algo import NSCLAlgo
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from nscl_predict import NSCLPredict as npredict
from pyvis.network import Network

import itertools
from itertools import islice


# test = [None, None, 2, None, None]
# exists = any(x != None for x in test)
# input(exists)


# dt = datetime.now().isoformat(timespec="minutes")
# input(dt)

# from networkx.drawing.nx_agraph import graphviz_layout

pathdiv = "/" if os.name == "posix" else "\\"
# pathdiv = "/" if os.name == "posix" else "\\"


def clear():
    if os.name == "posix":
        return os.system("clear")
    elif os.name == "nt":
        return os.system("cls")


themes = ["viridis", "mako_r", "flare"]

eng = NSCL.Engine()
# synapses = eng.network.synapses
# neurones = eng.network.neurones


def jprint(obj) -> None:
    js = json.dumps(obj, default=lambda x: x.__dict__, indent=4)
    print(js)


def reshape_trace(eng) -> list:
    print(len(eng.traces))
    timelength = len(eng.traces[-1])

    r = []
    for t in eng.traces:
        n = [0 for i in range(0, timelength)]
        for i, v in enumerate(t):
            n[i] = v
        r.append(n)

    # print("trace shape %s \n" % str(np.shape(r)))
    return r


def flatten(traces) -> list:
    return [item for sublist in traces for item in sublist]


# def heatmap(
#     result_path,
#     traces,
#     neurones,
#     figsize=(24, 12),
#     save=False,
#     ax=None,
#     theme="viridis",
# ):
#     print(" -heatplot")

#     my_colors = [(0.2, 0.3, 0.3), (0.4, 0.5, 0.4),
#                  (0.1, 0.7, 0), (0.1, 0.7, 0)]

#     tlength = len(traces)
#     arr_t = np.array(traces).T
#     # c = sns.color_palette("vlag", as_cmap=True)
#     # c = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=YlGnBu)

#     if save == True:
#         plt.figure(figsize=figsize)

#     heatplot = sns.heatmap(
#         arr_t,
#         # linewidth= (0.01 if tlength < 100 else 0.0),
#         linewidths=0.01,
#         yticklabels=neurones,
#         # cmap="YlGnBu",
#         cmap=theme,
#         # cmap=my_colors,
#         vmin=0,
#         vmax=1,
#         # linewidths=0.01,
#         square=True,
#         linecolor="#222",
#         annot_kws={"fontsize": 11},
#         # linecolor=(0.1,0.2,0.2),
#         xticklabels=10,
#         cbar=save,
#         ax=ax,
#     )

#     # colorbar = ax.collections[0].colorbar
#     # M=dt_tweet_cnt.max().max()
#     # colorbar.set_ticks([1/8*M,3/8*M,6/8*M])
#     # colorbar.set_ticklabels(['low','med','high'])

#     if save == True:
#         plt.tight_layout()
#         plt.savefig(result_path + pathdiv + r"Figure_1.png", dpi=120)
#         plt.clf()

#     return heatplot


# def lineplot(
#     result_path, data_plot, figsize=(15, 8), save=False, ax=None, theme="flare"
# ):
#     print(" -lineplot")
#     if save == True:
#         plt.figure(figsize=figsize)

#     lplot = sns.lineplot(
#         x="time", y="ncounts", data=data_plot, hue="ntype", ax=ax
#     )  # , palette=theme)

#     # for i in data_plot:
#     #     input(data_plot[i])

#     # lplot = plt.stackplot(x=range(0,60), y=data_plot, labels=["inputs", "generated", "total"])

#     if save == True:
#         plt.tight_layout()
#         plt.savefig(result_path + pathdiv + r"Figure_2.png", dpi=120)
#         plt.clf()

#     return lplot


# def networkx(result_path, synapses, figsize=(24, 12), save=False, ncolor="skyblue"):
#     print(" -networkplot")
#     plt.figure(figsize=figsize)
#     # Build a dataframe with your connections
#     df = pd.DataFrame(
#         {
#             "from": [synapses[s].rref for s in synapses],
#             "to": [synapses[s].fref for s in synapses],
#             "value": [synapses[s].wgt for s in synapses],
#         }
#     )

#     # Build your graph
#     G = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph())

#     # pos = graphviz_layout(G, prog='dot')

#     # Custom the nodes:
#     nx.draw(
#         G,
#         # pos,
#         with_labels=True,
#         node_color=ncolor,
#         node_size=1200,
#         edge_color=df["value"],
#         width=5.0,
#         alpha=0.9,
#         edge_cmap=plt.cm.Blues,
#         arrows=True,
#         # pos=nx.layout.planar_layout(G)
#     )
#     # plt.show()
#     # plt.figure(figsize=figsize)
#     if save == True:
#         plt.savefig(result_path + pathdiv + r"Figure_3.png", dpi=120)
#         plt.clf()

#     return G


# def networkx_pyvis(
#     result_path, synapses, figsize=(24, 12), save=False, ncolor="skyblue"
# ):
#     print(" -networkplot")
#     plt.figure(figsize=figsize)
#     # Build a dataframe with your connections
#     df = pd.DataFrame(
#         {
#             "from": [synapses[s].rref for s in synapses],
#             "to": [synapses[s].fref for s in synapses],
#             "value": [synapses[s].wgt for s in synapses],
#         }
#     )

#     # Build your graph
#     G = nx.from_pandas_edgelist(df, "from", "to", create_using=nx.DiGraph())

#     # pos = graphviz_layout(G, prog='dot')

#     # Custom the nodes:
#     nx.draw(
#         G,
#         # pos,
#         with_labels=True,
#         node_color=ncolor,
#         node_size=1200,
#         edge_color=df["value"],
#         width=5.0,
#         alpha=0.9,
#         edge_cmap=plt.cm.Blues,
#         arrows=True,
#         # pos=nx.layout.planar_layout(G)
#     )
#     # plt.show()
#     # plt.figure(figsize=figsize)
#     if save == True:
#         plt.savefig(result_path + pathdiv + r"Figure_3.png", dpi=120)
#         plt.clf()

#     #####################

#     net = Network(notebook=False, width=1600, height=900)
#     net.toggle_hide_edges_on_drag(False)
#     net.barnes_hut()
#     net.from_nx(nx.davis_southern_women_graph())
#     net.show("ex.html")

#     return G

def boxplot(data, title):

    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")

    # Load the example planets dataset
    # planets = sns.load_dataset("planets")
    # planets = data

    # pp.pprint(data)

    # Plot the orbital period with horizontal boxes
    sns.boxplot(x="interval", y="activity", data=data,
                whis=[0, 100], width=.6, palette="vlag").set(title=f'MIT Sensor Event Intevals ({title})')

    # Add in points to show each observation
    sns.stripplot(x="interval", y="activity", data=data,
                  size=4, color=".3", linewidth=0)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="Activities")
    ax.set(xlabel=f"Event intervals")
    sns.despine(trim=True, left=True)
    plt.tight_layout()
    plt.show()


def jsondump(result_path, fnameext, jdata):
    if not exists(result_path):
        os.mkdir(result_path)
    with open(result_path + pathdiv + fnameext, "w") as outfile:
        json.dump(jdata, outfile)


# def graphout(eng, flush=True):
#     tt = str(datetime.now().replace(microsecond=0)).replace(":", "_")
#     rpath = r"results%s%s" % (pathdiv, tt)

#     if os.name != "nt":
#         rpath = "results%s%s" % (pathdiv, tt)
#     eng.traces = reshape_trace(eng)

#     print("%s" % rpath)

#     if not os.path.exists("results"):
#         os.mkdir("results")

#     if not os.path.exists(rpath):
#         os.mkdir(rpath)

#     fig, ax = plt.subplots(2, 1, figsize=(24, 12))
#     df = pd.DataFrame(
#         {"time": eng.ntime, "ncounts": eng.ncounts, "ntype": eng.nmask})
#     sns.set_theme()
#     heatmap(rpath, eng.traces, eng.network.neurones, ax=ax[0])
#     lineplot(rpath, df, ax=ax[1])
#     plt.savefig(f"{rpath}{pathdiv}Figure_1-2.png", dpi=120)
#     plt.clf()
#     heatmap(rpath, eng.traces, eng.network.neurones, save=True)
#     lineplot(rpath, df, save=True)
#     networkx(rpath, eng.network.synapses, save=True)
#     networkx_pyvis(rpath, eng.network.synapses, save=True)

#     if flush == True:
#         eng.clear_traces()

def compileneuronegraph(fname="defparams.json", ticks=15, xres=8, yres=4):

    fig, axs = plt.subplots(1, 1)

    def neurone_profile(fname="defparams.json", start=0.30, ticks=ticks):
        with open(fname) as jsonfile:
            defparams = json.loads("\n".join(jsonfile.readlines()))
            start = defparams["FiringThreshold"] + 0.02

            x = [-4, -3, -2, -1]
            y = [start-19, start-17, start/2, start-13]

            x_t = [-2, -1]
            y_t = [
                defparams["FiringThreshold"],
                defparams["FiringThreshold"],
            ]

            x_z = [-2, -1]
            y_z = [
                defparams["ZeroingThreshold"],
                defparams["ZeroingThreshold"],
            ]

            x_b = [-2, -1]
            y_b = [
                defparams["BindingThreshold"],
                defparams["BindingThreshold"],
            ]

            value = start
            refractory = 0

            for i in range(0, ticks):
                # print(i, value)

                x.append(i)
                y.append(value)

                x_t.append(i)
                y_t.append(defparams["FiringThreshold"])

                x_z.append(i)
                y_z.append(defparams["ZeroingThreshold"])

                x_b.append(i)
                y_b.append(defparams["BindingThreshold"])

                if value < defparams["ZeroingThreshold"]:
                    value = 0
                elif value >= defparams["FiringThreshold"] and refractory == 0:
                    refractory = defparams["RefractoryPeriod"]
                    value = 1.0
                    # x.append(i)
                    # y.append(value)
                    value *= defparams["PostSpikeFactor"]
                else:
                    value *= defparams["DecayFactor"]

                if refractory > 0:
                    refractory = -1

            # x = np.linspace(-2, 12, num=11, endpoint=True)
            # y = np.cos(-x**2/9.0)
            # xnew = np.linspace(0, 15, num=41, endpoint=True)

            # f_cubic = interp1d(x, y, kind='cubic')
            # x, y = smooth(x, y)

            for i in range(0, len(x)):
                if y[i] < defparams["ZeroingThreshold"]:
                    y[i] = 0

            return (
                x,
                y,
                x_t,
                y_t,
                x_z,
                y_z,
                x_b,
                y_b,
                defparams
                # defparams["BindingThreshold"],
            )  # ,xnew, f_cubic(xnew))

    (nprof_x, nprof_y, nthres_x, nthres_y, nzero_x,
     nzero_y, binding_x, binding_y, defparams) = neurone_profile(fname, ticks=ticks)

    # plt.grid()
    xcoords = [x for x in range(-2, ticks)]
    active = [x-4 for x in xcoords if nprof_y[x]
              >= defparams["BindingThreshold"]]

    # print(nprof_x)
    # print(nprof_y)
    # print(active)
    # active.pop()
    print('active interval', len(active))
    input()
    inactive = [x for x in xcoords if x not in active]

    for xc in active:
        plt.axvline(x=xc, color="grey", linestyle="--", alpha=0.4)

    for xc in inactive:
        plt.axvline(x=xc, color="grey", linestyle="dotted", alpha=0.1)

    # for xc in xcoords:
    #     plt.axvline(x=xc, color="grey", linestyle="--")

    plt.plot(nthres_x, nthres_y, '--', label="Firing Threshold",
             color="black", linewidth=1.5)
    plt.plot(nzero_x, nzero_y, ':', label="Zeroing Threshold",
             color="black", linewidth=1.5)
    plt.plot(binding_x, binding_y, '-.', label="Binding Threshold",
             color="black", linewidth=1.5)
    plt.plot(nprof_x, nprof_y, label="Action Potential",
             color="black", linewidth=2)

    # plt.plot(
    #     [1],
    #     binding_threshold,
    #     marker="X",
    #     markersize=12,
    #     markeredgecolor="red",
    #     markerfacecolor="blue",
    # )

    plt.xlabel("Timestep")
    plt.ylabel("Neurone Potential")
    plt.legend(
        loc="best",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=1
    )
    plt.ylim(ymin=-0.1, ymax=1.1)
    plt.xlim([-2, ticks-1])

    # plt.show()
    fig.set_size_inches(xres, yres)
    plt.tight_layout()
    # plt.savefig(f"figures/neurone_profile.png", dpi=300)
    plt.show()


def stream(streamfile, trace=True):

    # text = "Top Cat! The most effectual Top Cat! Who’s intellectual close friends get to call him T.C., providing it’s with dignity. Top Cat! The indisputable leader of the gang. He’s the boss, he’s a pip, he’s the championship. He’s the most tip top, Top Cat."
    # txt_arr = text.lower().split(' ')
    # for i,v in enumerate(txt_arr):
    #     inputs[i] = [v]
    # input(inputs)

    filecontent = json.loads(
        open(f"dataset{pathdiv}{streamfile}.json", "r").read())

    interv = filecontent["interval"]
    inputs = filecontent["activity_stream"]

    temp = eng.tick
    eng.tick = 0

    maxit = min(len(inputs) - 1, interv)
    running = True

    r, e, a, p = ([], [], [], [])

    while running and eng.tick <= maxit:
        try:
            clear()

            print()
            print(" ###########################")
            print("     NSCL_python  t =", eng.tick)
            print(" ###########################")
            print()

            r, e, a, p = eng.algo(inputs[eng.tick], p, meta, trace)

            # if eng.tick == maxit:
            #     graphout(eng)

        except KeyboardInterrupt:
            running = False

    eng.tick += temp

    print("\n\n test streaming done.")
    print()


def normaliser(data, minn, maxx, scaling=1):
    try:
        return (data - minn) / (maxx - minn) * scaling
    except DivisionByZero:
        return 0


def csvstream(streamfile, metafile, trace=False, fname="default", iterations=1):
    # def take(n, iterable):
    #     # "Return first n items of the iterable as a list"
    #     return list(islice(iterable, n))

    data = {}

    metafile = open(metafile, "r")
    headers = metafile.readline().split(",")
    metadata = metafile.readlines()

    for line in metadata:

        row = line.split(",")
        # print(row)
        eng.meta[row[0]] = {
            "min": float(row[9]),
            "max": float(row[10]),
            "res": float(eng.network.params["DefaultEncoderResolution"]),
        }

        # eng.meta[row[0]] = {
        #     "min": eng.network.params["DefaultEncoderFloor"],
        #     "max": eng.network.params["DefaultEncoderCeiling"],
        #     "res": eng.network.params["DefaultEncoderResolution"],
        # }

    file = open(streamfile, "r")
    sensors = file.readline().split(",")[1:]
    rawdata = []
    rawcp = deepcopy(file.readlines())

    # print(len(rawdata))
    for i in range(iterations):
        rawdata += deepcopy(rawcp)
    # print(len(rawdata))
    # input()

    for i in range(len(rawdata)):
        # print(i, rawdata[i])
        entry = rawdata[i].split(',')
        entry[0] = str(i)
        rawdata[i] = ",".join(entry)

        # input(entry)

    # print(len(rawdata), rawdata[-1])
    # pp.pprint(rawdata[595: 615])

    # for line in rawdata:
    #     sline = line.replace("\n", "").split(",")
    #     for i in range(1, 1 + len(sensors)):
    #         if sline[i] != "": ## filters sensors
    #             sline[i] = f"{sensors[i-1]}~{sline[i]}"
    #     data[int(sline[0])] = [x for x in sline[1:] if x != ""]

    for line in rawdata:
        try:
            sline = line.replace("\n", "").split(",")
            for i in range(1, 1 + len(sensors)):
                if sline[i] != "":  # filters sensors
                    name = sensors[i - 1]
                    value = float(sline[i])

                    if (
                        search("current", name)
                        or search("humidity", name)
                        or search("temperature", name)
                    ):
                        sline[i] = ""
                    else:
                        maxx = eng.network.params["DefaultEncoderCeiling"]
                        minn = eng.network.params["DefaultEncoderFloor"]
                        res = eng.network.params["DefaultEncoderResolution"]

                        if name in eng.meta:
                            maxx = eng.meta[name]["max"]
                            minn = eng.meta[name]["min"]
                            res = eng.meta[name]["res"]

                        newval = math.floor(normaliser(value, minn, maxx, res))
                        # input(f"normaliser({value},{minn},{maxx},{res}) = {newval}")
                        sline[i] = f"{name}~{newval}-{newval+1}"
            data[int(sline[0])] = [x for x in sline[1:] if x != ""]
        except:
            print(line)

    start = int(rawdata[0].split(",")[0])
    end = int(rawdata[-1].split(",")[0])

    del rawdata

    # for i, v in enumerate(eng.meta):
    #     print(i, v["min"], v["max"], v["res"])

    print(eng.network.hash_id)
    print(start, end)
    save_range = [x for x in range(
        start + 1, end, int(round((end - start) / 20, 1)))]
    print(save_range)

    input("csv dataset loaded, now processing [enter] ")

    temp = eng.tick
    eng.tick = start
    maxit = end

    # maxit = min(len(inputs) - 1, interv)
    running = True
    # netmeta = open(f"{fname}.netmeta", "w+")

    starttime = datetime.now().isoformat(timespec="minutes")
    next_prompt = True

    r, e, a, p = ([], [], [], [])

    while running and eng.tick <= maxit:

        # print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # if not skip:
        #     for n in eng.network.neurones:
        #         print(f"{n}", end=" ")
        #     print()
        #     for n in eng.network.neurones:
        #         print(f"{eng.network.neurones[n].potential:0.2f}", end=" ")
        #     print()

        try:
            if eng.prune == 0:
                eng.prune = eng.network.params["PruneInterval"]
            eng.prune -= 1

            if eng.tick % 5000 == 0 or eng.tick == maxit or eng.prune == 0:
                clear()

                print()
                print(" ###########################")
                print(
                    f"     NSCL_python \n streming file {fname} at t(eng) {eng.tick}")
                print(f"hashid = {eng.network.hash_id}")
                print(f"start = {starttime}")
                print(f"saverange = {save_range}")
                print(
                    f"progress = {(eng.tick - start) / (end - start) * 100 : .1f}%")
                print(f"neurones = {len(eng.network.neurones)}")
                print(f"synapses = {len(eng.network.synapses)}")
                print(f"bindings = {eng.network.params['BindingCount']}")
                print(f"proplevel = {eng.network.params['PropagationLevels']}")
                print(
                    f"encres = {eng.network.params['DefaultEncoderResolution']}")
                print(f"npruned = {len(eng.npruned)}")
                print(f"spruned = {len(eng.spruned)}")
                print(f"prune_ctick = {eng.prune}")
                print(" ###########################")
                print()

            # print('\n', eng.tick)

            if eng.tick not in data:
                # print('Algo2 empty')
                r, e, a, p = eng.algo([], p)
                # print('input', [])
            else:
                # prevs = []
                # try:
                #     prevs = data[eng.tick-1]
                # except:
                #     pass

                # pp.pprint(data[eng.tick])
                # input()
                # input(data[eng.tick])

                # print('Algo2 previous')
                # print('algo')
                # print(data[eng.tick])
                # input()
                r, e, a, p = eng.algo(data[eng.tick], p)
                # print('input', data[eng.tick-1])

            # pp.pprint(eng.network.neurones.keys())

            if next_prompt and input("[enter] for next step ('r' for no-prompt): ") == 'r':
                next_prompt = False

            # if eng.tick == maxit and trace == True:
            #     graphout(eng)

            if eng.tick in save_range:
                save_range.remove(eng.tick)
                # os.mkdir(f'state{pathdiv}{fname}')
                # eng.save_state(f'{fname}{pathdiv}{eng.tick}')
                eng.save_state(f"{fname}_{eng.tick}")

            if eng.tick == maxit:
                input("csv stream done! [enter]")

        except KeyboardInterrupt:
            running = False

        # if input(f"\n{eng.tick-1}") == "s" and skip == False:
        #     skip = True

    # netmeta.close()
    eng.tick += temp

    print("\n csv streaming done.")
    print()


def jstream_test(filename):
    jfp = open(filename, 'r')
    return json.load(jfp)


def load_data(streamfile, metafile, trace=False, fname="default", exclude_values=["", "0"], exclude_headers=["current", "humidity", "temperature"]):
    # def take(n, iterable):
    #     # "Return first n items of the iterable as a list"
    #     return list(islice(iterable, n))

    data = {}

    metafile = open(metafile, "r")
    headers = metafile.readline().split(",")
    metadata = metafile.readlines()

    for line in metadata:

        row = line.split(",")
        eng.meta[row[0]] = {
            "min": float(row[9]),
            "max": float(row[10]),
            "res": float(eng.network.params["DefaultEncoderResolution"])
        }

    file = open(streamfile, "r")
    sensors = file.readline().split(",")[1:]
    rawdata = file.readlines()

    for line in rawdata:
        sline = line.replace("\n", "").split(",")
        for i in range(1, 1 + len(sensors)):
            if sline[i] != "":  # filters sensors
                name = sensors[i - 1]
                value = float(sline[i])

                exclude = all(n != None for n in [
                              search(x, name) for x in exclude_headers])

                if (exclude):
                    sline[i] = ""
                else:
                    maxx = eng.network.params["DefaultEncoderCeiling"]
                    minn = eng.network.params["DefaultEncoderFloor"]
                    res = eng.network.params["DefaultEncoderResolution"]

                    if name in eng.meta:
                        maxx = eng.meta[name]["max"]
                        minn = eng.meta[name]["min"]
                        res = eng.meta[name]["res"]

                    newval = math.floor(normaliser(value, minn, maxx, res))
                    # input(f"normaliser({value},{minn},{maxx},{res}) = {newval}")
                    sline[i] = f"{name}~{newval}-{newval+1}"

        data[int(sline[0])] = [x for x in sline[1:] if x != ""]

    del rawdata
    return data


def feed_trace_old(eng, tfirst, data, ticks=[], p_ticks=0, pot_threshold=0.8, reset_potentials=True):

    binding_window, refractory_period, reverberating, reverb_window = eng.network.check_params(
        prompt_fix=False)
    tick = tfirst

    if ticks == []:
        ticks = [t for t in range(max(data.keys()))]

    htraces = {}
    tscores = {}
    datainputs = {}
    refract_suppress = {}

    if reset_potentials:
        eng.reset_potentials()

    def neurones_levels(name):
        total = {}
        for lvls in eng.network.neurones[name].heirarcs.keys():
            if lvls not in total:
                total[lvls] = len(eng.network.neurones[name].heirarcs[lvls])
        return total

    def score(potential, ncount, heirarcs):
        scores = {}

        for lvls in heirarcs:
            for hns in heirarcs[lvls]:
                if hns not in scores:
                    scores[f"{hns}@{lvls+1}"] = potential / ncount
                    # scoring based on composites potential over number of connected synapses.

        return scores

    for t in ticks:

        # print('#####################################')
        activated = []

        # refract_suppress[t]

        for n in refract_suppress:
            refract_suppress[n] -= 1
            if refract_suppress[n] == 0:
                del refract_suppress[n]

        if t not in data.keys():
            r, e, activated = eng.algo([])
            # print(f"{tick} data", [])
        else:
            # print(f"{tick} data", data[t])
            datainputs[t] = data[t]
            r, e, activated = eng.algo(data[t])

        for n in activated:
            if n not in refract_suppress.keys():
                refract_suppress[n] = refractory_period

        # print(activated)
        actives = eng.get_actives(pot_threshold)
        htraces[tick] = []
        tscores[tick] = {}

        for n in actives:

            ncounts = sum(neurones_levels(n[0]).values())
            scores = score(n[1], ncounts, eng.network.neurones[n[0]].heirarcs)
            entry = (n[1], n[0], scores)
            htraces[tick].append(entry)

            for s in scores:
                if s not in tscores[tick]:
                    tscores[tick][s] = 0
                tscores[tick][s] += scores[s]

        tscores[tick] = [(x, tscores[tick][x]) for x in tscores[tick] if x.split(
            '@')[0] not in refract_suppress.keys()]
        tscores[tick].sort(reverse=True, key=lambda x: x[1])
        htraces[tick].sort(reverse=True, key=lambda x: x[0])

        # pp.pprint(htraces[t])
        # pp.pprint(tscores[t])
        # input()

        tick += 1

    for t in range(p_ticks):

        # print('#####################################')

        r, e, a = eng.algo([])
        # print(f"{tick} ptick", [])

        actives = eng.get_actives(pot_threshold)
        htraces[tick] = []
        tscores[tick] = {}

        for n in actives:

            ncounts = sum(neurones_levels(n[0]).values())
            scores = score(n[1], ncounts, eng.network.neurones[n[0]].heirarcs)
            entry = (n[1], n[0], scores)
            htraces[tick].append(entry)

            for s in scores:
                if s not in tscores[tick]:
                    tscores[tick][s] = 0
                tscores[tick][s] += scores[s]

        tscores[tick] = [(x, tscores[tick][x]) for x in tscores[tick]]
        tscores[tick].sort(reverse=True, key=lambda x: x[1])
        htraces[tick].sort(reverse=True, key=lambda x: x[0])

        # pp.pprint(htraces[t])
        # pp.pprint(tscores[tick])
        # input()

        tick += 1

    # pp.pprint(htraces)

    return tscores, htraces, datainputs


def feed_trace(eng, tfirst, data, ticks=[], p_ticks=0, reset_potentials=True):

    binding_window, refractory_period, reverberating, reverb_window = eng.network.check_params(
        prompt_fix=False)

    tick = tfirst

    if ticks == []:
        ticks = [t for t in range(max(data.keys()))]

    pscores = {}
    datainputs = {}
    templist = []

    if reset_potentials:
        eng.reset_potentials()

    # propagate & capture
    for t in ticks:

        if t not in data.keys():
            r, e, a = eng.algo([])
            for i in a:
                templist.append((tick, i, eng.network.neurones[i].potential))
        else:
            datainputs[t] = data[t]
            r, e, a = eng.algo(data[t])
            for i in a:
                templist.append((tick, i, eng.network.neurones[i].potential))
        tick += 1

    for t in range(p_ticks):

        # curractives = [
        #     x for x in templist if 'CMP' not in x[1] and x[0] == tick-1]
        # curractives.sort(key=lambda x: x[2], reverse=True)
        # window = [x for x in templist if x[0] > tick - 3]
        # window.sort(key=lambda x: x[2], reverse=True)

        # print('\n\n', tick)
        # print('current actives')
        # pp.pprint(curractives)
        # print('window actives')
        # pp.pprint(window)
        # input()

        # included
        r, e, a = eng.algo([])
        for i in a:
            templist.append((tick, i, eng.network.neurones[i].potential))
        tick += 1

        # pass

    # pp.pprint(templist)
    # input()

    for i in templist:
        # print(i[0], ' prime for ', i[1],
        #       npredict.primeProductWeights(i[1], eng))
        # input()

        if 'CMP' not in i[1]:
            if i[0] not in pscores:
                pscores[i[0]] = {}
            if i[1] not in pscores[i[0]]:
                pscores[i[0]][i[1]] = round(i[2], 4)
            else:
                pscores[i[0]][i[1]] += round(i[2], 4)
        else:
            s = npredict.primeProductWeights(i[1], eng)
            stotal = i[2]
            # print('s is ', s, ' total is ', i[2])

            for j in s:
                # print('t is ', i[0], 'offset is ', s[j][1])
                # k = s[j][1]+i[0]
                try:
                    if s[j][1]+i[0] not in pscores:
                        pscores[s[j][1] + i[0]] = {}
                    if j not in pscores[s[j][1]+i[0]]:
                        pscores[s[j][1]+i[0]][j] = s[j][0] * stotal
                    else:
                        # print(pscores[s[j][1]+i[0]][j])
                        pscores[s[j][1]+i[0]][j] += s[j][0] * stotal
                except:
                    continue

        # pp.pprint(pscores)
        # input()

    vals = [x for x in pscores]
    lowest = min(vals)
    highest = max(vals)
    # input(f'{lowest} {highest}')

    for i in range(lowest, highest):
        if i not in pscores:
            pscores[i] = {}

    return pscores, datainputs


def graph_distinct_predictions(pdistinct):

    input("graph distinct")
    pp.pprint(pdistinct)
    input()

    # glue = sns.load_dataset("glue").pivot("Model", "Task", "Score")
    # print(glue)
    # print('glue loaded')

    my_colors = [(0.2, 0.3, 0.3), (0.4, 0.5, 0.4),
                 (0.1, 0.7, 0), (0.1, 0.7, 0)]

    pdistarray = np.zeros((20, len(pdistinct)))

    counter = 0
    start = min([int(x) for x in pdistinct.keys()])
    for t in pdistinct:
        mx = max(pdistinct[t].values())
        # input(f"{t} {mx}")
        for s in pdistinct[t]:
            # input(f"{t} {s} {pdistinct[t][s]}")
            sidx = int(s.split("~")[0].replace("F", "1").replace("S", "0"))
            pdistarray[sidx][counter] = pdistinct[t][s] / mx
        counter += 1

    xdata = [x for x in range(start, start + len(pdistarray[0]))]

    # pdistarray = pdistarray[::-1]

    # pdistinct = [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4],
    #              [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]

    # tlength = len(pdistarray)
    arr_t = np.array(pdistarray)
    # c = sns.color_palette("vlag", as_cmap=True)
    # c = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=YlGnBu)

    sns.heatmap(
        arr_t,
        # linewidth= (0.01 if tlength < 100 else 0.0),
        linewidths=0.01,
        # cmap="YlGnBu",
        # cmap=my_colors,
        vmin=0,
        vmax=1,
        # xmin=19,
        # linewidths=0.01,
        square=True,
        linecolor="#222",
        annot_kws={"fontsize": 11},
        # linecolor=(0.1,0.2,0.2),
        xticklabels=xdata,
        cbar=save,
    )

    plt.gca().invert_yaxis()

    # colorbar = ax.collections[0].colorbar
    # M=dt_tweet_cnt.max().max()
    # colorbar.set_ticks([1/8*M,3/8*M,6/8*M])
    # colorbar.set_ticklabels(['low','med','high'])

    # if save == True:
    #     plt.tight_layout()
    #     plt.savefig(result_path + pathdiv + r"Figure_1.png", dpi=120)
    #     plt.clf()

    # sns.heatmap(glue)
    plt.show()


# converts json dictionary to algo-ready array
def jstream_to_algodict(jdict, tpad=0):
    data = {}

    lowest = 0
    highest = 0

    # if tpad < 0:

    # for activity in jdict:
    #     for instance in jdict[activity]:
    #         for sensor in instance:

    #             if lowest == 0 or lowest > sensor[1]:
    #                 lowest = sensor[1]
    #             if highest == 0 or highest < sensor[2]:
    #                 highest =sensor[2]

    #             print(sensor, lowest, highest)

    for activity in jdict:
        for instance in jdict[activity]:
            for sensor in instance:

                # if lowest == 0 or lowest > sensor[1]:
                #     lowest = sensor[1]
                # if highest == 0 or highest < sensor[2]:
                #     highest = sensor[2]

                # print(sensor, lowest, highest)
                # input()

                if sensor[1] not in data:
                    data[sensor[1]] = []
                data[sensor[1]].append(f'S{sensor[0]}~1')

                if sensor[2] not in data:
                    data[sensor[2]] = []
                data[sensor[2]].append(f'S{sensor[0]}~0')

    lowest = min(data.keys())
    highest = max(data.keys())

    print("lowest", lowest, "highest", highest)
    input()

    # merged timestamp
    if tpad >= 0:
        data.clear()
        # tcounter = 0
        for activity in jdict:
            for instance in jdict[activity]:
                for sensor in instance:
                    tact = sensor[1] - lowest
                    if tact not in data:
                        data[tact] = []
                    data[tact].append(f'S{sensor[0]}~1')

                    tdeact = sensor[2] - lowest
                    if tdeact not in data:
                        data[tdeact] = []
                    data[tdeact].append(f'S{sensor[0]}~0')
                # tcounter += 1

    pp.pprint(data)


######### main prog starts #########
args = sys.argv[1:]

# print(" ".join(args))

pp = pprint.PrettyPrinter(indent=4)

init = False
verbose = False

while True:
    # try:
    if init == False:
        print("\n\n########### NSCL (Python) ###########\n")
        print(f" version: experimental/non-optimised")
        print(f" os.name: {os.name}")
        print(f" os.pid: {os.getpid()}")
        print("")

        # subprocess.call(f'top -p {os.getpid()}', shell=True)
        # os.system(f"top -p {os.getpid()}")
        # Popen('bash')

    # try:
    if init == True:
        if os.name == "posix":
            command = input(
                f"\033[1m\033[96m {os.getpid()}: NSCL [{eng.tick}]> \u001b[0m\033[1m"
            ).split(" ")
        else:
            command = input(f" {os.getpid()}: NSCL [{eng.tick}]> ").split(" ")
    else:
        init = True
        command = args

    if len(command) == 0:
        continue

    if command[0] in ["clear", "cls", "clr"]:
        clear()

    if command[0] in ["param", "set"]:
        if command[1] in ["verb", "verbose"]:
            verbose = bool(command[2])
            print(f"verbose={verbose}")

    if command[0] in ["check", "checkparams"]:
        binding_window, refractory_period, reverberating, reverb_window = eng.network.check_params()
        print("Binding Window =", binding_window)
        print("Current Refractoryperiod =", refractory_period)
        print("Reverberating Firing =", reverberating)
        print("Suggested Refractoryperiod:", reverb_window)

    if command[0] == "params":
        for p in eng.network.params:
            print(f"{p} {eng.network.params[p]}")

    if command[0] == "stream":
        print(" streaming test dataset as input - %s" % command[1])
        stream(command[1])

    if command[0] == "csvstream_traced":
        print(" streaming csv dataset as input - %s" % command[1])
        csvstream(command[1], command[2], True, command[3])

    if command[0] == "csvstream":
        eng.network.check_params()
        epoch = int(input('Epoch: '))
        print(" streaming csv dataset as input - %s" % (command[1]))
        csvstream(command[1], command[2], False, command[3], epoch)

    if command[0] == "jstream_test":
        data = jstream_test('activities_dict.json')
        keys = data.keys()

        # input(data)

        # include_keys = [
        #     'Doing laundry',
        #     'Watching TV',
        #     'Preparing breakfast',
        # ]

        include_keys = [x for x in data.keys()]

        # print('timelogs')
        # pprint.pprint(include_keys)
        # input('\n')

        included_data = {}

        # watching_tv = data['Watching TV']
        # watching_tv_lengths = []

        # lowest = 0
        # highest = 0

        intervals_onoff = {'activity': [], 'interval': []}
        intervals_on = {'activity': [], 'interval': []}
        intervals_off = {'activity': [], 'interval': []}
        sensors = []
        timelogs_all = []

        for include in include_keys:
            for instance in data[include]:

                lengths = instance[-1][1] - instance[0][1]
                instance = sorted(instance, key=lambda x: x[1], reverse=False)

                if include not in included_data:
                    included_data[include] = []

                included_data[include].append(instance)

            timelogs = []

            for instance in data[include]:
                for entry in instance:
                    timelogs.append((entry[1], int(entry[0]), 1))
                    timelogs.append((entry[2], int(entry[0]), 0))

            timelogs_all.extend(timelogs)
            timelogs = sorted(timelogs, key=lambda x: x[0], reverse=False)
            earliest = timelogs[0][0]
            latest = timelogs[-1][0]
            timelogs_on = [x for x in timelogs if x[2] == 1]
            timelogs_off = [x for x in timelogs if x[2] == 0]
            sensors.extend(item[1] for item in timelogs)

            print('timelogs')
            print(len(timelogs), len(timelogs_all))
            input()

            for i in range(len(timelogs)-1):
                intervals_onoff['activity'].append(include)
                intervals_onoff['interval'].append(
                    timelogs[i+1][0] - timelogs[i][0])

            for i in range(len(timelogs_on)-1):
                intervals_on['activity'].append(include)
                intervals_on['interval'].append(
                    timelogs_on[i+1][0] - timelogs_on[i][0])

            for i in range(len(timelogs_off)-1):
                intervals_off['activity'].append(include)
                intervals_off['interval'].append(
                    timelogs_off[i+1][0] - timelogs_off[i][0])

        # pp.pprint(included_data)
        # input()

        sensors = list(set(sensors))
        sensors.sort()
        print('sensors', sensors)

        df = pd.DataFrame.from_dict(intervals_onoff)
        boxplot(df, 'On/Off')

        medians = []
        for act_unique in df['activity'].unique():
            print(act_unique, df.loc[df['activity'] == act_unique]['interval'].mean(),
                  df.loc[df['activity'] == act_unique]['interval'].median())
            medians.append(
                df.loc[df['activity'] == act_unique]['interval'].median())

        print('med_avg', sum(medians)/len(medians))
        print('med_max', max(medians))
        print('t_earliest', earliest)
        print('t_latest', latest)
        # input()

        timelogs_all = sorted(timelogs_all, key=lambda x: x[0], reverse=False)
        earliest = timelogs_all[0][0]
        latest = timelogs_all[-1][0]

        tdata = {}
        csv_content = []
        csv_content_unpadded = []
        headers = [str(s) for s in sensors]
        headers.insert(0, 'unix_time')
        csv_content.append(','.join(headers)+',\n')
        csv_content_unpadded.append(','.join(headers)+',\n')

        for t in timelogs_all:
            if t[0]-earliest not in tdata:
                tdata[t[0]-earliest] = []
            tdata[t[0]-earliest].append((t[1], t[2]))

        print(timelogs_all)

        for t in range(0, latest-earliest+60):
            x = ['' for x in headers]
            x[0] = str(t)

            # print('t = ', t, end='')

            skip = True
            if t in tdata:
                for it in tdata[t]:
                    if it[1] == 1 or it[1] == 0:  # (on, on/off, off)
                        idx = headers.index(str(it[0]))
                        x[idx] = str(it[1])
                        skip = False

                    # print('adding', x)
            csv_content.append(','.join(x)+',\n')
            if not skip:
                csv_content_unpadded.append(','.join(x)+',\n')

        # print('done')

        fs = open(f'dataset/MIT_{len(include_keys)}S_ONOFF.csv', 'w')
        fs.writelines(csv_content)
        fs.close()
        fs = open(f'dataset/MIT_{len(include_keys)}S_ONOFF_UPADDED.csv', 'w')
        fs.writelines(csv_content_unpadded)
        fs.close()

        # META

        meta_headers = "sensor,records,elapsed,unix_oldest,unix_newest,oldest,newest,minimum,maximum,min2,max2,min3,max3,\n"

        meta_line = "name,,,,,,,,,min,max,,,"
        meta_content = [meta_headers]

        for h in headers:
            if "F" in h:
                meta_content.append(
                    meta_line.replace("name", h).replace(
                        "min", "0").replace("max", "1024")
                    + "\n"
                )
            elif "S" in h:
                meta_content.append(
                    meta_line.replace("name", h).replace(
                        "min", "0").replace("max", "1") + "\n"
                )
            else:
                meta_content.append(
                    meta_line.replace("name", h).replace(
                        "min", "0").replace("max", "1") + "\n"
                )

        print(meta_content)

        f = open(f'dataset/MIT_{len(include_keys)}S_meta.csv', 'w')
        # f = open(f"meta_MIT{f_tagname}.csv", "w")
        f.writelines(meta_content)

        # print(csv_content)

        # input()

        # df = pd.DataFrame.from_dict(intervals_on)
        # boxplot(df, 'On Only')

        # df = pd.DataFrame.from_dict(intervals_off)
        # boxplot(df, 'Off Only')

        # pp.pprint(included_data)

        # jstream_to_algodict(included_data)

        # pp.pprint(included_data)

    if command[0] == "infprogj":
        def inputDef(prompt, defval, casting):
            x = input(prompt)
            try:
                return casting(x)
            except:
                return defval

        print("infer progressive")

        data = jstream_test('activities_dict.json')

        # print('see activities_dict.json for parameters')
        # predict_activity = input('activity to predict: ')
        # activity_number = int(input('activity number: '))
        # priories = int(input("first priories sequence (length): "))

        # include_keys = [
        #     'Doing laundry',
        #     # 'Watching TV',
        #     # 'Preparing breakfast',
        # ]

        include_keys = [x for x in data.keys()]

        # pactivity = data[predict_activity][activity_number]
        pactivity = []
        for activity in include_keys:
            pactivity.extend([item for sublist in data[activity]
                             for item in sublist])

        # pp.pprint(pactivity)
        # input()

        # print(pactivity)
        # input()

        # file_data = f"./dataset/{command[1]}"
        # file_meta = f"./dataset/old/meta_{command[1]}_{command[2]}.csv"

        # alldata = load_data(file_data, file_meta)

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False

        # first = int(input("first priories: "))
        # last = int(input("last row in file (stream to): "))
        pticks = inputDef("infer length after stream (def 20): ",
                          eng.network.params['InferLength'], int)
        refract = inputDef("infer-refractory period (def 4): ",
                           eng.network.params['InferRefractoryWindow'], int)
        divergence = inputDef("divergence count (def 4): ",
                              eng.network.params['InferDivergence'], int)

        results = {}

        pactivity_stream = {}

        for entry in pactivity:
            if entry[1] not in pactivity_stream:
                pactivity_stream[entry[1]] = []
            pactivity_stream[entry[1]].append(f'{entry[0]}~1-2')
            # if entry[2] not in pactivity_stream:
            #     pactivity_stream[entry[2]] = []
            # pactivity_stream[entry[2]].append(f'{entry[0]}~0-1')

        def unpad(act_stream):
            result = {}
            keys = [x for x in act_stream.keys()]
            keys.sort()
            for i in range(0, len(keys)):
                result[i] = act_stream[keys[i]]
            return result

        pactivity_stream = unpad(pactivity_stream)
        pp.pprint(pactivity_stream)
        input()

        earliest = min(pactivity_stream.keys())
        last = max(pactivity_stream.keys())
        for i in range(last, last+60):
            if i not in pactivity_stream:
                pactivity_stream[i] = []

        # dataCompare = {}

        # for t in range(first, last):
        #     data[t] = deepcopy(alldata[t])

        # for t in range(first, last+pticks):
        #     results[t] = deepcopy(alldata[t])

        # del alldata

        # priories = deepcopy(data)

        # for i in len()
        # print(last)
        # pp.pprint(pactivity_stream)
        # input()

        pact_stream = list(pactivity_stream.items())
        # input(pact_stream)
        pact_stream.sort(key=lambda x: x[0])

        accpredictions = {}
        accinputs = {}

        # count, top prediction, ctx prediction, out of context
        counts = [0, 0, 0, 0]

        for rindex in range(earliest+1, last+1):
            # data = dict(itertools.islice(pactivity_stream.items(), rindex))
            # print(pact_stream)
            # j = pact_stream[earliest-earliest: rindex-earliest]
            # print(j)
            data = {}

            for i in range(earliest, rindex):
                if i in pactivity_stream and pactivity_stream[i] != []:
                    data[i] = pactivity_stream[i]

            pscores, datainputs, tick, rea, predictions, pdistinct, ctxacc, neweng = npredict.infer_unfolding(eng, data, pticks, reset_potentials,
                                                                                                              propagation_count=20, refractoryperiod=refract, divergence=divergence)

            # eng = neweng

            # print("Accuracy")
            # print("Accuracy")

            # print("Params")
            # pp.pprint(eng.network.params)

            # print
            

            clear()

            print('####################### START ##########################')
            print(earliest, rindex, last, '---', earliest -
                  earliest, rindex-earliest, last-earliest)


            high = 0
            try:
                high = max(pscores.keys())
            except:
                pass

            predictions = {high:{}}
            try:
                predictions = {high: pscores[high]}  # pscore at future timestep
            except:
                pass

            # filter to only highest divergence
            predictions_list = [(x, predictions[list(predictions.keys())[0]][x])
                                for x in predictions[list(predictions.keys())[0]].keys()]
            predictions_list = sorted(predictions_list, key=lambda x: x[1], reverse=True)[:divergence]
            predictions = dict(predictions_list)
            predictions = {high:predictions}

            pdata = [x for x in list(data.items()) if x[0] >= rindex - 200]
            pdata = [j for sub in [x[1] for x in pdata] for j in sub]


            if True:
                # print("\nAccInputs")
                # pp.pprint(data)

                # print('\nWPScores')
                # pp.pprint(pred)

                # if len(accinputs.values()) != 0:
                #     print('\nWPrevAccInputs')
                #     pp.pprint(list(accinputs.values())[-1])

                # print("\nPscores")
                # pp.pprint(pscores)

                if len(accpredictions.values()) != 0:
                    print('\nPrevPredictions')
                    pp.pprint(list(accpredictions.values())[-1])

                print('\nWAccInputs')
                pp.pprint(pdata[-1])

                # potentials = [(x,eng.network.neurones[x].potential) for x in pdata]

                # print('\Potentials')
                # pp.pprint(potentials)
                print("\nPredictions")
                pp.pprint(predictions)

                # input()

            accpredictions[rindex] = {}
            if len(predictions.keys()) != 0:
                accpredictions[rindex] = predictions

            if len(pdata) != 0:
                accinputs[rindex] = pdata

            if len(accinputs.values()) >= 2:
                if str(list(accinputs.values())[-2]) != str(list(accinputs.values())[-1]):
                    counts[0] += 1

                    prev_pscore = list(accpredictions.values())[-2]
                    inp = pdata[-1]
                    inp_prevv = pdata[-2]
                    t = [x for x in prev_pscore.keys()][0]
                    prev_pscore = prev_pscore[t]

                    if inp in prev_pscore:
                        counts[2] += 1

                    try:
                        prev_pscore.pop(inp_prevv)
                        highest_value = max(prev_pscore, key=prev_pscore.get)
                        if inp == highest_value:
                            counts[1] += 1
                    except:
                        pass

            print()
            print(counts)

            print('####################### END ##########################')
            # input()

        input()

        # save = input("\nsave results in filename (nosave, leave empty): ")

        # if save != "":
        #     jsondump("feedtraces2", f"{save}.json", {
        #              "params": eng.network.params, "meta": eng.meta, "datainputs": priories, "predictions": predictions, "accinputs": datainputs, "pscores_prod": pscores})

    if command[0] == "infprogs":
        def inputDef(prompt, defval, casting):
            x = input(prompt)
            if x == '':
                return defval
            try:
                return casting(x)
            except:
                return defval

        print("infer progressive")

        file = inputDef(
            "data to load", "./dataset/old/dataset_sin_S10F10.csv", str)
        metafile = inputDef(
            "meta to load", "./dataset/old/meta_sin_S10F10.csv", str)

        pactivity_stream = {}

        metafile = open(metafile, "r")
        metadata = metafile.readlines()[1:]

        for line in metadata:

            row = line.split(",")
            # print(row)
            eng.meta[row[0]] = {
                "min": float(row[9]),
                "max": float(row[10]),
                "res": float(eng.network.params["DefaultEncoderResolution"]),
            }

        fs = open(file, 'r')
        data = fs.readlines()
        data = [x.replace('\n', '') for x in data]
        header = data[0].split(',')

        for line in data[1:]:
            entry = line.split(',')
            pactivity_stream[int(entry[0])] = []
            for idx, value in enumerate(entry):
                if value != '' and idx != 0:
                    sensor = header[idx]
                    name = eng.encoded_neu_name(sensor, int(value))
                    # input(name)
                    pactivity_stream[int(entry[0])].append(name)
            # input(pactivity_stream)

        # pp.pprint(pactivity_stream)
        # input()

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False

        # first = int(input("first priories: "))
        # last = int(input("last row in file (stream to): "))
        pticks = inputDef("infer length after stream (def 20): ",
                          eng.network.params['InferLength'], int)
        refract = inputDef("infer-refractory period (def 4): ",
                           eng.network.params['InferRefractoryWindow'], int)
        divergence = inputDef("divergence count (def 4): ",
                              eng.network.params['InferDivergence'], int)

        results = {}

        # pactivity_stream = {}

        # for entry in pactivity:
        #     if entry[1] not in pactivity_stream:
        #         pactivity_stream[entry[1]] = []
        #     pactivity_stream[entry[1]].append(f'{entry[0]}~1-2')
        # if entry[2] not in pactivity_stream:
        #     pactivity_stream[entry[2]] = []
        # pactivity_stream[entry[2]].append(f'{entry[0]}~0-1')

        def unpad(act_stream):
            result = {}
            keys = [x for x in act_stream.keys()]
            keys.sort()
            for i in range(0, len(keys)):
                result[i] = act_stream[keys[i]]
            return result

        # pactivity_stream = unpad(pactivity_stream)
        pp.pprint(pactivity_stream)
        input()

        earliest = min(pactivity_stream.keys())
        last = max(pactivity_stream.keys())
        for i in range(last, last+60):
            if i not in pactivity_stream:
                pactivity_stream[i] = []

        # dataCompare = {}

        # for t in range(first, last):
        #     data[t] = deepcopy(alldata[t])

        # for t in range(first, last+pticks):
        #     results[t] = deepcopy(alldata[t])

        # del alldata

        # priories = deepcopy(data)

        # for i in len()
        # print(last)
        # pp.pprint(pactivity_stream)
        # input()

        pact_stream = list(pactivity_stream.items())
        # input(pact_stream)
        pact_stream.sort(key=lambda x: x[0])

        accpredictions = {}
        accinputs = {}

        # count, top prediction, ctx prediction, out of context
        counts = [0, 0, 0, 0]

        for rindex in range(earliest+1, last+1):
            # data = dict(itertools.islice(pactivity_stream.items(), rindex))
            # print(pact_stream)
            # j = pact_stream[earliest-earliest: rindex-earliest]
            # print(j)
            data = {}

            for i in range(earliest, rindex):
                if i in pactivity_stream and pactivity_stream[i] != []:
                    data[i] = pactivity_stream[i]

            pscores, datainputs, tick, rea, predictions, pdistinct, ctxacc, neweng = npredict.infer_unfolding(eng, data, pticks, reset_potentials,
                                                                                                              propagation_count=20, refractoryperiod=refract, divergence=divergence)

            # eng = neweng

            # print("Accuracy")
            # print("Accuracy")

            # print("Params")
            # pp.pprint(eng.network.params)

            # print("\nInputs")
            # pp.pprint(pactivity_stream)
            clear()

            print('####################### START ##########################')
            print(earliest, rindex, last, '---', earliest -
                  earliest, rindex-earliest, last-earliest)

            # print("\nPScores")
            # pp.pprint(pscores)
            high = max(pscores.keys())
            # predictions
            # predictions = {high:Counter(pscores[high])+ Counter(pscores[high-1])}
            predictions = {high: pscores[high]}
            # pscores = list(pscores.items())

            pdata = [x for x in list(data.items()) if x[0] >= rindex - 200]
            pdata = [j for sub in [x[1] for x in pdata] for j in sub]

            # pred = [x for x in list(pscores.items()) if x[0] >= rindex -200 and x[0] <= rindex+200]
            # pred =  [pscores[tt] for tt in [t for t in pscores if t >= rindex -200 and t <= rindex+200] if tt in pscores]
            # for sett in pred:
            #     for key in sett.keys():
            #         if key not in predictions and key not in pdata:
            #             predictions[key] = 0.0
            #         if key not in pdata:
            #             predictions[key] += sett[key]

            if True:
                # print("\nAccInputs")
                # pp.pprint(data)

                # print('\nWPScores')
                # pp.pprint(pred)

                # if len(accinputs.values()) != 0:
                #     print('\nWPrevAccInputs')
                #     pp.pprint(list(accinputs.values())[-1])

                # print("\nPscores")
                # pp.pprint(pscores)

                if len(accpredictions.values()) != 0:
                    print('\nPrevPredictions')
                    pp.pprint(list(accpredictions.values())[-1])

                print('\nWAccInputs')
                pp.pprint(pdata[-1])

                # potentials = [(x,eng.network.neurones[x].potential) for x in pdata]

                # print('\Potentials')
                # pp.pprint(potentials)
                print("\nPredictions")
                pp.pprint(predictions)

                # input()

            accpredictions[rindex] = {}
            if len(predictions.keys()) != 0:
                accpredictions[rindex] = predictions

            if len(pdata) != 0:
                accinputs[rindex] = pdata

            if len(accinputs.values()) >= 2:
                if str(list(accinputs.values())[-2]) != str(list(accinputs.values())[-1]):
                    counts[0] += 1

                    prev_pscore = list(accpredictions.values())[-2]
                    inp = pdata[-1]
                    inp_prevv = pdata[-2]
                    t = [x for x in prev_pscore.keys()][0]
                    prev_pscore = prev_pscore[t]

                    if inp in prev_pscore:
                        counts[2] += 1

                    try:
                        prev_pscore.pop(inp_prevv)
                        highest_value = max(prev_pscore, key=prev_pscore.get)
                        if inp == highest_value:
                            counts[1] += 1
                    except:
                        pass

            print()
            print(counts)

            print('####################### END ##########################')
            # input()

        input()

        # save = input("\nsave results in filename (nosave, leave empty): ")

        # if save != "":
        #     jsondump("feedtraces2", f"{save}.json", {
        #              "params": eng.network.params, "meta": eng.meta, "datainputs": priories, "predictions": predictions, "accinputs": datainputs, "pscores_prod": pscores})

    if command[0] in ["tracepaths", "trace", "traces", "paths", "path"]:
        limits = float(command[1])
        inp = command[2].split(",")
        print(f" NSCL.trace(limits={limits})")
        print(inp)
        pp.pprint(npredict.trace_paths(eng, inp, limits, verbose=True))

    if command[0] == "active":
        active = eng.get_actives()
        pp.pprint(active)

    if command[0] == "thresholds":
        active = eng.get_actives(float(command[1]))
        pp.pprint(active)

    if command[0] == "backtrace":
        propslvl = eng.network.params["PropagationLevels"]  # float(command[1])
        neurone = command[1]
        print(f" NSCL.back_trace()")
        print("propslvl", propslvl)
        print("composite", neurone)
        pp.pprint(npredict.back_trace(propslvl, neurone))

    if command[0] == "infunfold":
        def inputDef(prompt, defval, casting):
            x = input(prompt)
            try:
                return casting(x)
            except:
                return defval

        print("infer unfold")

        file_data = f"./dataset/old/dataset_{command[1]}_{command[2]}.csv"
        file_meta = f"./dataset/old/meta_{command[1]}_{command[2]}.csv"

        alldata = load_data(file_data, file_meta)

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False
        first = int(input("first row in file (stream from): "))
        last = int(input("last row in file (stream to): "))
        pticks = inputDef("infer length after stream (def 20): ",
                          eng.network.params['InferLength'], int)
        refract = inputDef("infer-refractory period (def 2): ",
                           eng.network.params['InferRefractoryWindow'], int)
        divergence = inputDef("divergence count (def 3): ",
                              eng.network.params['InferDivergence'], int)

        data = {}
        dataCompare = {}

        for t in range(first, last):
            data[t] = deepcopy(alldata[t])

        for t in range(first, last+pticks):
            dataCompare[t] = deepcopy(alldata[t])

        del alldata

        priories = deepcopy(data)
        pscores, datainputs, tick, rea, predictions, pdistinct = npredict.infer_unfolding(deepcopy(eng), data, pticks, reset_potentials,
                                                                                          propagation_count=20, refractoryperiod=refract, divergence=divergence)

        for t in range(last, last+pticks-2):
            try:
                sc = [p[1] for d in dataCompare[t]
                      for p in predictions[t] if p[0] == d]
                # sc = [x[1] for x in predictions[t] if x[0] == dataCompare[t][0]][0]
                print(t, dataCompare[t], sc, predictions[t])
            except:
                print('error (context lost), no t-entry at', t)

        print("Accuracy")
        # print("Accuracy")

        print("Params")
        pp.pprint(eng.network.params)

        print("\nPScores")
        pp.pprint(pscores)

        print("\nInputs")
        pp.pprint(priories)

        print("\nAccInputs")
        pp.pprint(datainputs)

        print("\nPredictions")
        pp.pprint(predictions)

        print("\npDistinct")
        pp.pprint(pdistinct)

        save = input("\nsave results in filename (nosave, leave empty): ")

        if save != "":
            jsondump("feedtraces2", f"{save}.json", {
                     "params": eng.network.params, "meta": eng.meta, "datainputs": priories, "predictions": predictions, "accinputs": datainputs, "pscores_prod": pdistinct})

        graph_distinct_predictions(pdistinct)

    # if command[0] == "infprogr":
    #     def inputDef(prompt, defval, casting):
    #         x = input(prompt)
    #         try:
    #             return casting(x)
    #         except:
    #             return defval

    #     print("infer progressive")

    #     file_data = f"./dataset/old/dataset_{command[1]}_{command[2]}.csv"
    #     file_meta = f"./dataset/old/meta_{command[1]}_{command[2]}.csv"

    #     alldata = load_data(file_data, file_meta)

    #     reset_potentials = True if input(
    #         "reset potentials (y/n): ") == 'y' else False
    #     first = int(input("first row in file (stream from): "))
    #     last = int(input("last row in file (stream to): "))
    #     pticks = inputDef("infer length after stream (def 20): ",
    #                       eng.network.params['InferLength'], int)
    #     refract = inputDef("infer-refractory period (def 4): ",
    #                        eng.network.params['InferRefractoryWindow'], int)
    #     divergence = inputDef("divergence count (def 4): ",
    #                           eng.network.params['InferDivergence'], int)

    #     data = {}
    #     dataCompare = {}

    #     for t in range(first, last):
    #         data[t] = deepcopy(alldata[t])

    #     for t in range(first, last+pticks):
    #         dataCompare[t] = deepcopy(alldata[t])

    #     del alldata

    #     pprint.pprint(data)
    #     input()

    #     priories = deepcopy(data)
    #     pscores, datainputs, tick, rea, predictions, pdistinct = npredict.infer_unfolding(deepcopy(eng), data, pticks, reset_potentials,
    #                                                                                       propagation_count=20, refractoryperiod=refract, divergence=divergence)

    #     for t in range(last, last+pticks-2):
    #         try:
    #             sc = [p[1] for d in dataCompare[t]
    #                   for p in predictions[t] if p[0] == d]
    #             # sc = [x[1] for x in predictions[t] if x[0] == dataCompare[t][0]][0]
    #             print(t, dataCompare[t], sc, predictions[t])
    #         except:
    #             print('error (context lost), no t-entry at', t)

    #     print("Accuracy")
    #     # print("Accuracy")

    #     print("Params")
    #     pp.pprint(eng.network.params)

    #     print("\nPScores")
    #     pp.pprint(pscores)

    #     print("\nInputs")
    #     pp.pprint(priories)

    #     print("\nAccInputs")
    #     pp.pprint(datainputs)

    #     print("\nPredictions")
    #     pp.pprint(predictions)

    #     save = input("\nsave results in filename (nosave, leave empty): ")

    #     if save != "":
    #         jsondump("feedtraces2", f"{save}.json", {
    #                  "params": eng.network.params, "meta": eng.meta, "datainputs": priories, "predictions": predictions, "accinputs": datainputs, "pscores_prod": pscores})

    if command[0] == "infunfoldacc":
        def inputDef(prompt, defval, casting):
            x = input(prompt)
            try:
                return casting(x)
            except:
                return defval

        print("infer unfold accuracy")
        input()

        file_data = f"./dataset/dataset_{command[1]}_{command[2]}.csv"
        file_meta = f"./dataset/meta_{command[1]}_{command[2]}.csv"

        alldata = load_data(file_data, file_meta)

        reset_potentials = True if input(
            "reset potentials (y/n): ") == 'y' else False
        first = int(input("first row in file (stream from): "))
        last = int(input("last row in file (stream to): "))
        pticks = inputDef("infer length after stream (def 20): ",
                          eng.network.params['InferLength'], int)
        refract = inputDef("infer-refractory period (def 4): ",
                           eng.network.params['InferRefractoryWindow'], int)
        divergence = inputDef("divergence count (def 8): ",
                              eng.network.params['InferDivergence'], int)

        data = {}
        dataCompare = {}

        for t in range(first, last):
            data[t] = deepcopy(alldata[t])

        for t in range(first, last+pticks):
            dataCompare[t] = deepcopy(alldata[t])

        del alldata

        accres = []
        for div in range(1, divergence+1):
            for ref in range(1, refract+1):

                # pscores={}
                # datainputs={}
                # tick = 0

                # priories = deepcopy(data)

                pscores, datainputs, tick, reap, predictions, pdistinct = npredict.infer_unfolding(deepcopy(eng), deepcopy(data), deepcopy(pticks), reset_potentials,
                                                                                                   propagation_count=20, refractoryperiod=ref, divergence=div)

                # print(f"D: {div} R: {ref}")
                # pp.pprint(predictions)
                # input()

                potacc = 0.0
                ctxloss = 0
                ctxcount = 0
                ctxcorr = 0
                ctxpotacc = -1
                for t in range(last, last+pticks-2):
                    ctxcount += 1
                    try:
                        # check if items in datacompare are also in predictions list
                        sc = [p[1] for d in dataCompare[t]
                              for p in pdistinct[t] if p[0] == d][0]
                        # sc = [x[1] for x in predictions[t] if x[0] == dataCompare[t][0]][0]
                        potacc += sc
                        # print(t, dataCompare[t], sc, predictions[t])

                        dcomp = dataCompare[t]
                        pcomp = [p[0]
                                 for p in pdistinct[t]][:len(dataCompare[t])]

                        ctxcorr += len(list(set(dcomp) & set(pcomp)))
                        # if ctxcorr > 0:
                        #     print("correct context", t, ctxcorr, dcomp, pcomp)
                        #     input()

                    except:
                        ctxloss += 1
                        # print('error (context lost), no t-entry at', t)

                del pscores
                del datainputs
                del tick
                del predictions
                del pdistinct

                try:
                    ctxpotacc = potacc * (1 - ctxloss/ctxcount)
                except:
                    pass

                print(
                    f"Diverg: {div} Refrac: {ref} PotTtl: {potacc:0.2f} CtxLoss: {ctxloss/ctxcount:0.2f} CtxAcc: {ctxcorr} CtxPotDist: {ctxpotacc:0.2f}")
                accres.append((div, ref, potacc, ctxloss / ctxcount,
                              ctxcorr, ctxpotacc))

        accres.sort(key=lambda x: x[3], reverse=True)
        accres.sort(key=lambda x: x[4], reverse=True)
        pp.pprint(accres[:10])
        tops = deepcopy(accres[:10])

        accres.sort(key=lambda x: x[4], reverse=True)
        accres.sort(key=lambda x: x[3], reverse=True)
        pp.pprint(accres[:10])
        tops.extend(deepcopy(accres[:10]))

        tag = input(
            "Save results to ctxacc.csv with tag: (leave empty, no save)? ")

        if tag != "":
            composeorder = ""
            if eng.network.params['ComposeByCompositeFirst']:
                composeorder += "C"
            else:
                composeorder += "P"
            if eng.network.params['ComposeByPotentialFirst']:
                composeorder += "P"
            else:
                composeorder += "A"
            tag = f"B{eng.network.params['BindingCount']}L{eng.network.params['PropagationLevels']}_{composeorder}_{tag}"

            try:
                fs = open("./states/ctxacc.csv", "r")
                fs.close()
            except:
                fs = open("./states/ctxacc.csv", "w+")
                fs.write("Datetime,NetworkHashID,PredictionTag,PtRange,Binding,Levels,ComposeByPotentialsFirst,ComposeByCompositionFirst,Divergence,Refractory,PotTtl,CtxLoss,CtxAcc,CtxPotDist,\n")
                fs.close()

            ctxaccfile = open("./states/ctxacc.csv", "a")

            for entry in tops:
                pfile = f"{command[1]}_{command[2]}.csv"
                ctxaccfile.write(
                    f"{datetime.now()},{eng.network.hash_id},{tag},{first}->{last}+{pticks},{eng.network.params['BindingCount']},{eng.network.params['PropagationLevels']},{eng.network.params['ComposeByPotentialFirst']},{eng.network.params['ComposeByCompositeFirst']},{entry[0]},{entry[1]},{round(entry[2],4)},{round(entry[3],4)},{round(entry[4],4)},{round(entry[5],4)},\n")

            ctxaccfile.close()
            print(" [ ctxacc.csv appended ] ")

    # if command[0] == "infprogr":
    #     def inputDef(prompt, defval, casting):
    #         x = input(prompt)
    #         try:
    #             return casting(x)
    #         except:
    #             return defval

    #     print("infer progressive")

    #     file_data = f"./dataset/dataset_{command[1]}_{command[2]}.csv"
    #     file_meta = f"./dataset/meta_{command[1]}_{command[2]}.csv"

    #     alldata = load_data(file_data, file_meta)

    #     reset_potentials = True if input(
    #         "reset potentials (y/n): ") == 'y' else False
    #     first = int(input("first row in file (stream from): "))
    #     last = int(input("last row in file (stream to): "))
    #     # pticks = inputDef("infer length after stream (def 20): ",
    #     #                   eng.network.params['InferLength'], int)
    #     refract = inputDef("infer-refractory period (def 4): ",
    #                        eng.network.params['InferRefractoryWindow'], int)
    #     divergence = inputDef("divergence count (def 8): ",
    #                           eng.network.params['InferDivergence'], int)

    #     data = {}
    #     dataCompare = {}

    #     for t in range(first, last):
    #         data[t] = deepcopy(alldata[t])

    #     for t in range(first, last+pticks):
    #         dataCompare[t] = deepcopy(alldata[t])

    #     del alldata

    #     accres = []
    #     for div in range(1, divergence+1):
    #         for ref in range(1, refract+1):
    #             pscores, datainputs, tick, predictions = npredict.feed_forward_progressive(deepcopy(eng), deepcopy(data), reset_potentials,
    #                                                                                        propagation_count=20, refractoryperiod=ref, divergence=div)

    #             potacc = 0.0
    #             ctxloss = 0
    #             ctxcount = 0
    #             ctxcorr = 0
    #             ctxpotacc = -1

    #             for t in range(last, last+pticks-2):
    #                 ctxcount += 1
    #                 try:
    #                     # check if items in datacompare are also in predictions list
    #                     sc = [p[1] for d in dataCompare[t]
    #                           for p in predictions[t] if p[0] == d][0]

    #                     potacc += sc
    #                     dcomp = dataCompare[t]
    #                     pcomp = [p[0]
    #                              for p in predictions[t]][:len(dataCompare[t])]

    #                     ctxcorr += len(list(set(dcomp) & set(pcomp)))
    #                 except:
    #                     ctxloss += 1

    #             del pscores
    #             del datainputs
    #             del tick
    #             del predictions

    #             try:
    #                 ctxpotacc = potacc * (1 - ctxloss/ctxcount)
    #             except:
    #                 pass

    #             print(
    #                 f"Diverg: {div} Refrac: {ref} PotTtl: {potacc:0.2f} CtxLoss: {ctxloss/ctxcount:0.2f} CtxAcc: {ctxcorr} CtxPotDist: {ctxpotacc:0.2f}")
    #             accres.append((div, ref, potacc, ctxloss / ctxcount,
    #                           ctxcorr, ctxpotacc))

    #     accres.sort(key=lambda x: x[3], reverse=True)
    #     accres.sort(key=lambda x: x[4], reverse=True)
    #     pp.pprint(accres[:10])
    #     tops = deepcopy(accres[:10])

    #     accres.sort(key=lambda x: x[4], reverse=True)
    #     accres.sort(key=lambda x: x[3], reverse=True)
    #     pp.pprint(accres[:10])
    #     tops.extend(deepcopy(accres[:10]))

    #     tag = input(
    #         "Save results to ctxacc.csv with tag: (leave empty, no save)? ")

    #     if tag != "":
    #         composeorder = ""
    #         if eng.network.params['ComposeByCompositeFirst']:
    #             composeorder += "C"
    #         else:
    #             composeorder += "P"
    #         if eng.network.params['ComposeByPotentialFirst']:
    #             composeorder += "P"
    #         else:
    #             composeorder += "A"
    #         tag = f"B{eng.network.params['BindingCount']}L{eng.network.params['PropagationLevels']}_{composeorder}_{tag}"

    #         try:
    #             fs = open("./states/ctxacc.csv", "r")
    #             fs.close()
    #         except:
    #             fs = open("./states/ctxacc.csv", "w+")
    #             fs.write("Datetime,NetworkHashID,PredictionTag,PtRange,Binding,Levels,ComposeByPotentialsFirst,ComposeByCompositionFirst,Divergence,Refractory,PotTtl,CtxLoss,CtxAcc,CtxPotDist,\n")
    #             fs.close()

    #         ctxaccfile = open("./states/ctxacc.csv", "a")

    #         for entry in tops:
    #             pfile = f"{command[1]}_{command[2]}.csv"
    #             ctxaccfile.write(
    #                 f"{datetime.now()},{eng.network.hash_id},{tag},{first}->{last}+{pticks},{eng.network.params['BindingCount']},{eng.network.params['PropagationLevels']},{eng.network.params['ComposeByPotentialFirst']},{eng.network.params['ComposeByCompositeFirst']},{entry[0]},{entry[1]},{round(entry[2],4)},{round(entry[3],4)},{round(entry[4],4)},{round(entry[5],4)},\n")

    #         ctxaccfile.close()
    #         print(" [ ctxacc.csv appended ] ")

    if command[0] == "trace":
        print("tracing")

        for i in range(int(command[4]), int(command[5])):

            file_data = f"./dataset/dataset_{command[1]}_{command[2]}.csv"
            file_meta = f"./dataset/meta_{command[1]}_{command[2]}.csv"

            data = load_data(file_data, file_meta)

            reset_potentials = True
            # first = int(input("first row in file (stream from): "))
            # last = int(input("last row in file (stream to): "))
            # pticks = int(input("post ticks after stream: "))

            first = int(command[4])
            last = int(command[5])
            pticks = 10

            ticks = [x for x in range(first, i)]

            # for i in range(first, last):
            #     print(i, data[i])

            # input(f"\nStream Length: {len(data)}  \nStream Range: {ticks}\n")

            tscores, datainputs = feed_trace(deepcopy(
                eng), first, data, ticks=ticks, p_ticks=pticks, reset_potentials=reset_potentials)

            print("Params")
            pp.pprint(eng.network.params)

            print("\nInputs")
            pp.pprint(datainputs)

            print("\nPScores")
            pp.pprint(tscores)

            save = f"{command[3]}_{command[4]}_{i}"

            if save != "":
                jsondump("feedtraces2", f"{save}.json", {
                    "params": eng.network.params, "meta": eng.meta, "datainputs": datainputs, "tscores_prod": tscores})

            # data = data[first:last]
            print('done')

    if command[0] == "feed":
        feed = [x for x in command[1].split(",") if x != ""]
        eng.algo(feed)

    if command[0] == "prune":
        ncount = len(eng.network.neurones)
        scount = len(eng.network.synapses)
        npcount = len(eng.npruned)
        spcount = len(eng.spruned)
        print(f" NSCL.prune()")
        eng.prune_network()
        print(f" ncount {ncount} -> {len(eng.network.neurones)}")
        print(f" scount {scount} -> {len(eng.network.synapses)}")
        print(f" npcount {npcount} -> {len(eng.npruned)}")
        print(f" spcount {spcount} -> {len(eng.spruned)}")

    if command[0] in ["potsyn", "ptsyn", "struct", "network", "ls"]:
        eng.potsyns()
        print()

    if command[0] == "new":
        confirm = input
        if input(" new network? (y/n)") == "y":
            del eng
            eng = NSCL.Engine()
            print("new net")

    # if command[0] == "graphout":
    #     print(" exporting graphs")
    #     graphout(eng)

    if command[0] == "save":
        # if len(command) > 1:
        print(f" Savestate({command[1]})")
        eng.save_state(command[1])
        # else:
        # composeorder = ""
        # if eng.network.params['ComposeByCompositeFirst']:
        #     composeorder += "C"
        # else:
        #     composeorder += "P"
        # if eng.network.params['ComposeByPotentialFirst']:
        #     composeorder += "P"
        # else:
        #     composeorder += "A"
        # eng.save_state(f"B{eng.network.params['BindingCount']}L{eng.network.params['PropagationLevels']}_{composeorder}_{command[1]}")

    if command[0] == "load":
        print(f" Loadstate({command[1]})")
        del eng
        eng = NSCL.Engine()
        print(f" memsize={eng.load_state(command[1])}")

    if command[0] == "memsize":
        print(f" memsize={eng.size_stat()}")
        # print(eng.size_stat())

    if command[0] == "metalist":
        print(f" metalist={eng.meta}")
        # print(eng.size_stat())

    if command[0] == "avg_wgt_r":
        print(f" neurone {command[1]} = {eng.network.avg_wgt_r(command[1])}")

    if command[0] == "avg_wgt_f":
        print(f" neurone {command[1]} = {eng.network.avg_wgt_f(command[1])}")

    if command[0] == "info":
        clear()

        print()
        print(" ########################### ")
        print(f"     NSCL_python ")
        print()
        print(f"tick = {eng.tick}")
        print(f"hashid = {eng.network.hash_id}")
        # print(f"progress = {(eng.tick - start) / (end - start) * 100 : .1f}%")
        print(f"neurones = {len(eng.network.neurones)}")
        print(f"synapses = {len(eng.network.synapses)}")
        print(f"bindings = {eng.network.params['BindingCount']}")
        print(f"PropagationLevels = {eng.network.params['PropagationLevels']}")
        print(f"npruned = {len(eng.npruned)}")
        print(f"spruned = {len(eng.spruned)}")
        print(f"prune_ctick = {eng.prune}")
        print()
        eng.network.check_params()
        print()
        pp.pprint(eng.network.params)
        print()
        print(" ########################### ")
        print()

    if command[0] in ["tick", "pass", "next"]:
        r, e = eng.algo([], {})
        print(" reinf %s " % r)
        print(" errs %s " % e)

    if command[0] == "test":
        del eng
        eng = NSCL.Engine()
        print(f" memsize={eng.load_state('DSB2L2_S10F10_DW2')}")
        testcode = NSCL.Test.CompName()
        pp.pprint(NSCL.Test.primeProductWeights(testcode.name, eng))

    if command[0] == 'nprofile':
        compileneuronegraph(ticks=int(command[1]), xres=5, yres=5)

    if command[0] == "exit":
        sys.exit(0)
    # except Exception as e:
    #     print(str(e))


# cd /mnt/Data/Dropbox/PhD Stuff/Najiy/sourcecodes/nscl-python
