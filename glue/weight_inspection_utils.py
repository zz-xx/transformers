"""
Random weight inspection utilities to evaluate whether models change a lot
with STILTs or not. Not really used anywhere...
"""

import numpy as np


def get_np_w(module):
    return module.weight.detach().numpy()


def svd_perc_at_n(arr, n=100):
    s = np.linalg.svd(arr, compute_uv=False)
    return s[:n].sum() / s.sum()


def plot_layers(model):
    layers = model.bert.encoder.layer
    for i, layer in enumerate(layers[:5]):
        print(f"Layer {i}:")
        att = layer.attention
        self = att.self
        q = get_np_w(self.query)
        print(f"  query: {svd_perc_at_n (q) * 100:.5f}%")
        k = get_np_w(self.key)
        print(f"  key: {svd_perc_at_n (k) * 100:.1f}%")
        v = get_np_w(self.value)
        print(f"  value: {svd_perc_at_n (v) * 100:.1f}%")
        att_out = get_np_w(att.output.dense)
        print(f"  att_out: {svd_perc_at_n (att_out) * 100:.1f}%")
        inter = get_np_w(layer.intermediate.dense)
        print(f"  inter: {svd_perc_at_n (inter) * 100:.1f}%")
        fin_out = get_np_w(layer.output.dense)
        print(f"  fin_out: {svd_perc_at_n (fin_out) * 100:.1f}%")


def distance_between_weights(m1, m2, ord=2):
    m1_layers, m2_layers = m1.bert.encoder.layer, m2.bert.encoder.layer
    dists_q = []
    dists_k = []
    dists_v = []
    dists_a = []
    dists_i = []
    dists_o = []
    for i, (m1_layer, m2_layer) in enumerate(zip(m1_layers, m2_layers)):
        m1_att, m2_att = m1_layer.attention.self, m2_layer.attention.self
        m1_q, m2_q = get_np_w(m1_att.query), get_np_w(m2_att.query)
        m1_k, m2_k = get_np_w(m1_att.key), get_np_w(m2_att.key)
        m1_v, m2_v = get_np_w(m1_att.value), get_np_w(m2_att.value)
        m1_a, m2_a = (
            get_np_w(m1_layer.attention.output.dense),
            get_np_w(m2_layer.attention.output.dense),
        )
        m1_int, m2_int = (
            get_np_w(m1_layer.intermediate.dense),
            get_np_w(m2_layer.intermediate.dense),
        )
        m1_o, m2_o = (
            get_np_w(m1_layer.output.dense),
            get_np_w(m2_layer.output.dense),
        )

        dists_q.append(np.linalg.norm(m1_q - m2_q, ord=ord))
        dists_k.append(np.linalg.norm(m1_k - m2_k, ord=ord))
        dists_v.append(np.linalg.norm(m1_v - m2_v, ord=ord))
        dists_a.append(np.linalg.norm(m1_a - m2_a, ord=ord))
        dists_i.append(np.linalg.norm(m1_int - m2_int, ord=ord))
        dists_o.append(np.linalg.norm(m1_o - m2_o, ord=ord))
    dists = {
        "query": dists_q,
        "key": dists_k,
        "value": dists_v,
        "att_out": dists_a,
        "intermediate": dists_i,
        "final_out": dists_o,
    }
    return dists

# plt.figure(figsize=(10, 10))
# for l, layers in dists_orig_mnli.items():
#     plt.plot(layers, label=l)
# plt.title("L2 distance between original BERT and STILTs MNLI version per layer")
# plt.xlabel("Layer")
# plt.ylabel("L2 norm")
# plt.legend()