"""Unified sequence-embedding architectures for the Poseigen suite."""

import numpy as np

import torch
import torch.nn as nn

import poseigen_trident.prongs as tp


class Mimic(nn.Module):
    """Combines sequence embedding types into one architecture."""

    def __init__(
        self,
        dim_i=(1, 249, 4),
        dim_f=(1, 1, 1),
        trunc=False,
        kE_k=15,
        kE_cf_m=100,
        kE_ck_base=4,
        kE_OneByOne=False,
        kE_ck_grouped=False,
        kE_freeze=False,
        PR_cf_m=2,
        PR_ck_base=5,
        PR_pool_func=nn.AvgPool2d,
        PR_pk_base=2,
        PR_pk_s=2,
        P_ck=50,
        PR_layer=None,
        PR_num_layers=1,
        PR_bi=True,
        O_mods=3,
        O_cf_pu_m=1,
        O_dropout=0.2,
        cf_ns=1,
        activations=nn.ReLU(),
        activation_f=None,
        batchnorm="before",
    ):
        super(Mimic, self).__init__()

        self.trunc = trunc

        kE_cf_f = kE_cf_m * dim_i[0]

        if (dim_i[1] - kE_k + 1) < dim_f[1]:
            kE_k = dim_i[1] - dim_f[1] + 1

        dim_i_ref = dim_i
        dim_kE_width = 1
        dim_kE_length = dim_i[1] - kE_k + 1
        dim_kE = (kE_cf_f, dim_kE_length, dim_kE_width)

        kE_Prong_args = {
            "dim_i": dim_i_ref,
            "dim_f": dim_kE,
            "mods": 0,
            "mods_ns": 0,
            "cf_i": None,
            "cf_ns": cf_ns,
            "ck_base": kE_ck_base,
            "doublestranded": False,
            "OneByOne": kE_OneByOne,
            "ck_grouped": kE_ck_grouped,
            "activations": activations,
            "activation_f": None,
            "batchnorm": batchnorm,
            "dropout": None,
            "bias": True,
            "out": False,
        }

        self.kE = nn.Sequential(*tp.Prong_X(**kE_Prong_args))

        if kE_freeze:
            for param in self.kE.parameters():
                param.requires_grad = False

        P_func = PR_pool_func

        if P_ck is None:
            P_ck = 0
        if P_ck == 0:
            self.P, P_length = (nn.Identity(), dim_kE_length)
        elif P_ck > 1:
            pad2add = 0
            self.P = P_func(
                (P_ck, 1),
                stride=(P_ck, 1),
                padding=(pad2add, 0),
                ceil_mode=True,
            )
            P_length = int(np.ceil(dim_kE_length / P_ck))
        else:
            self.P = P_func(
                (dim_kE_length, 1),
                stride=(1, 1),
                padding=(0, 0),
                ceil_mode=True,
            )
            P_length = 1

        self.rnn = False

        mid_f = int(PR_cf_m * dim_kE[0])
        dim_mid = (mid_f, 1, 1)

        if O_mods == 0:
            self.PR = nn.Identity()
            dim_mid = (dim_kE[0], P_length, 1)
            mid_f = dim_kE[0]

        elif PR_layer is None:
            skip_first_conv = P_length == dim_kE[1]
            dim_kE = (dim_kE[0], P_length, 1)

            PR_args = {
                "dim_i": dim_kE,
                "dim_f": dim_mid,
                "cf_ns": cf_ns,
                "ck_i": None,
                "skip_first_conv": skip_first_conv,
                "ck_base": PR_ck_base,
                "pk_base": PR_pk_base,
                "pk_s": PR_pk_s,
                "activations": activations,
                "batchnorm": batchnorm,
            }
            self.PR = nn.Sequential(*tp.Prong_W(**PR_args))

        elif PR_layer in [nn.GRU, nn.LSTM]:
            self.rnn = True
            hid_f = mid_f
            if PR_bi:
                hid_f = mid_f // 2
                mid_f = hid_f * 2
            dim_mid = (mid_f, 1, 1)
            self.PR_bi = PR_bi
            self.PR = PR_layer(
                input_size=dim_kE[0],
                hidden_size=int(hid_f),
                num_layers=PR_num_layers,
                bidirectional=PR_bi,
                batch_first=True,
            )
        elif PR_layer in [nn.Conv2d]:
            self.PR = nn.Conv2d(
                dim_kE[0],
                mid_f,
                kernel_size=(P_length, 1),
                bias=True,
                stride=(1, 1),
                padding=0,
            )

        else:
            dim_mid = (dim_kE[0], 1, 1)
            self.PR = PR_layer(embed_dim=dim_kE[0])
            mid_f = dim_kE[0]

        if trunc:
            self.O = nn.Identity()
        else:
            O_args = {
                "dim_i": dim_mid,
                "dim_f": dim_f,
                "mods": O_mods,
                "cf_ns": cf_ns,
                "cf_pu": O_cf_pu_m * mid_f,
                "activations": activations,
                "activation_f": activation_f,
                "batchnorm": batchnorm,
                "dropout": O_dropout,
                "out": True,
            }
            self.O = nn.Sequential(*tp.Prong_X(**O_args))

    def forward(self, x):
        x = self.kE(x)
        x = self.P(x)

        if self.rnn:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)

            rnn = self.PR
            if isinstance(rnn, nn.LSTM):
                _, (h_n, _) = rnn(x)
            elif isinstance(rnn, nn.GRU):
                _, h_n = rnn(x)

            num_directions = 2 if self.PR_bi else 1
            if num_directions == 1:
                x = h_n[-1]
            else:
                x = torch.cat([h_n[-2], h_n[-1]], dim=1)
            x = x.unsqueeze(-1).unsqueeze(-1)
        else:
            x = self.PR(x)

        return self.O(x)


def Reset_Mimic(mimic):
    lconvs, lprs, ldens = len(mimic.kE), len(mimic.PR), len(mimic.O)

    for i in np.arange(lconvs):
        if isinstance(mimic.kE[i], nn.Conv2d):
            mimic.kE[i].reset_parameters()

    if ldens > 1:
        for i in np.arange(ldens):
            if isinstance(mimic.O[i], nn.Conv2d):
                mimic.O[i].reset_parameters()

    if lprs > 1:
        for i in np.arange(lprs):
            if isinstance(mimic.PR[i], nn.Conv2d):
                mimic.PR[i].reset_parameters()
    elif lprs == 1:
        mimic.PR.reset_parameters()

    return mimic


class Dual_Mimic(nn.Module):
    def __init__(
        self,
        dim_f=(1, 1, 1),
        late_comb=True,
        A_dim_i=(1, 249, 4),
        A_kE_k=15,
        A_kE_cf_m=100,
        A_kE_ck_base=4,
        A_PR_ck_base=5,
        A_PR_pk_base=2,
        A_PR_pk_s=2,
        A_P_ck=None,
        B_dim_i=(1, 249, 4),
        B_kE_k=15,
        B_kE_cf_m=100,
        B_kE_ck_base=4,
        B_PR_ck_base=5,
        B_PR_pk_base=2,
        B_PR_pk_s=2,
        B_P_ck=None,
        PR_layer=None,
        PR_num_layers=1,
        PR_bi=True,
        cf_ns=1,
        kE_freeze=False,
        kE_OneByOne=False,
        PR_pool_func=nn.MaxPool2d,
        PR_cf_m=2,
        O_mods=3,
        O_cf_pu_m=2,
        O_dropout=0,
        activations=nn.ReLU(),
        activation_f=None,
        batchnorm="before",
    ):
        super(Dual_Mimic, self).__init__()

        trunc = not late_comb
        kE_ck_grouped = False

        mimic_shared_args = {
            "dim_f": dim_f,
            "trunc": trunc,
            "PR_layer": PR_layer,
            "PR_num_layers": PR_num_layers,
            "PR_bi": PR_bi,
            "cf_ns": cf_ns,
            "kE_freeze": kE_freeze,
            "kE_ck_grouped": kE_ck_grouped,
            "kE_OneByOne": kE_OneByOne,
            "PR_pool_func": PR_pool_func,
            "PR_cf_m": PR_cf_m,
            "O_mods": O_mods,
            "O_cf_pu_m": O_cf_pu_m,
            "O_dropout": O_dropout,
            "activations": activations,
            "batchnorm": batchnorm,
        }

        mimic_A_args = {
            "dim_i": A_dim_i,
            "kE_k": A_kE_k,
            "kE_cf_m": A_kE_cf_m,
            "kE_ck_base": A_kE_ck_base,
            "P_ck": A_P_ck,
            "PR_ck_base": A_PR_ck_base,
            "PR_pk_base": A_PR_pk_base,
            "PR_pk_s": A_PR_pk_s,
            **mimic_shared_args,
        }
        mimic_B_args = {
            "dim_i": B_dim_i,
            "kE_k": B_kE_k,
            "kE_cf_m": B_kE_cf_m,
            "kE_ck_base": B_kE_ck_base,
            "P_ck": B_P_ck,
            "PR_ck_base": B_PR_ck_base,
            "PR_pk_base": B_PR_pk_base,
            "PR_pk_s": B_PR_pk_s,
            **mimic_shared_args,
        }

        self.late_comb = late_comb

        if late_comb:
            self.ultraA = Mimic(**mimic_A_args)
            self.ultraB = Mimic(**mimic_B_args)
            self.O = nn.Identity()
        else:
            self.ultraA = Mimic(**mimic_A_args)
            self.ultraB = Mimic(**mimic_B_args)

            if PR_layer is not None:
                if PR_layer not in [nn.GRU, nn.LSTM, nn.Conv2d]:
                    PR_cf_m = 1

            A_kE_f = A_kE_cf_m * A_dim_i[0]
            A_mid_f = PR_cf_m * A_kE_f

            B_kE_f = B_kE_cf_m * B_dim_i[0]
            B_mid_f = PR_cf_m * B_kE_f

            if PR_layer in [nn.GRU, nn.LSTM] and PR_bi:
                hid_f = A_mid_f // 2
                A_mid_f = hid_f * 2

                hid_f = B_mid_f // 2
                B_mid_f = hid_f * 2

            AB_mid_f = int(A_mid_f + B_mid_f)
            dim_mid = (AB_mid_f, 1, 1)

            O_args = {
                "dim_i": dim_mid,
                "dim_f": dim_f,
                "mods": O_mods,
                "cf_ns": cf_ns,
                "cf_pu": int(O_cf_pu_m * AB_mid_f),
                "activations": activations,
                "batchnorm": batchnorm,
                "dropout": O_dropout,
                "out": True,
            }
            self.O = nn.Sequential(*tp.Prong_X(**O_args))

        self.FA = activation_f if activation_f is not None else nn.Identity()

    def forward(self, x1, x2):
        x1 = self.ultraA(x1)
        x2 = self.ultraB(x2)

        if self.late_comb:
            xcomb = x1 + x2
        else:
            xcomb = torch.cat([x1, x2], dim=1)

        xcomb = self.O(xcomb)
        return self.FA(xcomb)


def Reset_Dual_Mimic(dual_mimic):
    for mimic in [dual_mimic.ultraA, dual_mimic.ultraB]:
        Reset_Mimic(mimic)

    lex = len(dual_mimic.O)

    if lex > 1:
        for i in np.arange(lex):
            if isinstance(dual_mimic.O[i], nn.Conv2d):
                dual_mimic.O[i].reset_parameters()

        print("done reset mod")

    return dual_mimic
