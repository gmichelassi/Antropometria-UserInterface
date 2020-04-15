import numpy as np
from scipy.spatial import distance as dist

import logger

log = logger.getLogger(__file__)


# XXX colocar essa funcao num script utils
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# qtd total de distancias possiveis: 2278
# 68 pontos que combinados 2 a 2 (distancia), p or arranjo simples sem repeticao
# C(m,p) = m! / ((m-p)! * p!)
# C(68,2) = 68! / ((68 - 2)! * 2!) = 2278
def farkas(points, reference):
    # esta estrutura descreve as distancias baseadas no livro do Farkas, ou seja,
    # NAO possui todas as distancias possiveis.
    distances_eu = {}  # euclidian distance
    distances_mh = {}  # manhattan (cityblock) distance

    log.info("Calculating Farkas et al distances.")
    # Metade inferior da altura craniofacial FARKAS et al p 47
    distances_eu["enr_gn"] = (dist.euclidean((points["x_42"], points["y_42"]), (points["x_8"], points["y_8"]))) / \
                             reference["w-pixels"]
    distances_mh["enr_gn"] = (dist.cityblock((points["x_42"], points["y_42"]), (points["x_8"], points["y_8"]))) / \
                             reference["w-pixels"]
    # distancia do canto externo ao canto interno olho esquerdo.
    # As distancias em FARKAS et al. sao ex e en dos 2 lados porem coloquei indica-
    # cao de esquerda e direita
    distances_eu["exl_enl"] = (dist.euclidean((points["x_36"], points["y_36"]), (points["x_39"], points["y_39"]))) / \
                              reference["w-pixels"]
    distances_mh["exl_enl"] = (dist.cityblock((points["x_36"], points["y_36"]), (points["x_39"], points["y_39"]))) / \
                              reference["w-pixels"]
    # distancia do canto interno ao canto externo olho direito
    distances_eu["enr_exr"] = (dist.euclidean((points["x_42"], points["y_42"]), (points["x_45"], points["y_45"]))) / \
                              reference["w-pixels"]
    distances_mh["enr_exr"] = (dist.cityblock((points["x_42"], points["y_42"]), (points["x_45"], points["y_45"]))) / \
                              reference["w-pixels"]

    # distancia do canto externo ao canto externo olho esquerdo ao direito
    distances_eu["exl_exr"] = (dist.euclidean((points["x_36"], points["y_36"]), (points["x_45"], points["y_45"]))) / \
                              reference["w-pixels"]
    distances_eu["exl_exr"] = (dist.euclidean((points["x_36"], points["y_36"]), (points["x_45"], points["y_45"]))) / \
                              reference["w-pixels"]
    distances_mh["exl_exr"] = (dist.cityblock((points["x_36"], points["y_36"]), (points["x_45"], points["y_45"]))) / \
                              reference["w-pixels"]
    distances_mh["exl_exr"] = (dist.cityblock((points["x_36"], points["y_36"]), (points["x_45"], points["y_45"]))) / \
                              reference["w-pixels"]

    # distancia intercantal
    distances_eu["enl_enr"] = (dist.euclidean((points["x_39"], points["y_39"]), (points["x_42"], points["y_42"]))) / \
                              reference["w-pixels"]
    distances_mh["enl_enr"] = (dist.cityblock((points["x_39"], points["y_39"]), (points["x_42"], points["y_42"]))) / \
                              reference["w-pixels"]

    # distancia ponto entre os olhos ate o queixo
    distances_eu["n_gn"] = (dist.euclidean((points["x_8"], points["y_8"]), (points["x_27"], points["y_27"]))) / \
                           reference["w-pixels"]
    distances_mh["n_gn"] = (dist.cityblock((points["x_8"], points["y_8"]), (points["x_27"], points["y_27"]))) / \
                           reference["w-pixels"]

    # largura mandibula. FARKAS et al. pag 46
    # 'l' indica 'left' e 'r' indica 'right'
    distances_eu["gol_gor"] = (dist.euclidean((points["x_4"], points["y_4"]), (points["x_12"], points["y_12"]))) / \
                              reference["w-pixels"]
    distances_mh["gol_gor"] = (dist.cityblock((points["x_4"], points["y_4"]), (points["x_12"], points["y_12"]))) / \
                              reference["w-pixels"]

    # largura base do cranio. FARKAS et al. pag 46
    # 'l' indica 'left' e 'r' indica 'right'
    distances_eu["zyl_zyr"] = (dist.euclidean((points["x_2"], points["y_2"]), (points["x_14"], points["y_14"]))) / \
                              reference["w-pixels"]
    distances_mh["zyl_zyr"] = (dist.cityblock((points["x_2"], points["y_2"]), (points["x_14"], points["y_14"]))) / \
                              reference["w-pixels"]

    # comprimento nariz. Ponto entre o meio dos olhos e ponta do nariz
    distances_eu["n_prn"] = (dist.euclidean((points["x_27"], points["y_27"]), (points["x_32"], points["y_32"]))) / \
                            reference["w-pixels"]
    distances_mh["n_prn"] = (dist.cityblock((points["x_27"], points["y_27"]), (points["x_32"], points["y_32"]))) / \
                            reference["w-pixels"]

    # Distancia entre os pontos medianos da palpebra superior e inferior do olho esquerdo.
    # Calculo diferente devido a localizacao dos pontos de acordo com saida
    # e definicao em FARKAS et al
    distances_eu["psl_pil_1"] = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_41"], points["y_41"]))) / \
                                reference["w-pixels"]
    distances_eu["psl_pil_2"] = (dist.euclidean((points["x_38"], points["y_38"]), (points["x_40"], points["y_40"]))) / \
                                reference["w-pixels"]
    distances_mh["psl_pil_1"] = (dist.cityblock((points["x_37"], points["y_37"]), (points["x_41"], points["y_41"]))) / \
                                reference["w-pixels"]
    distances_mh["psl_pil_2"] = (dist.cityblock((points["x_38"], points["y_38"]), (points["x_40"], points["y_40"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano palpebra superior ao canto externo olho esquerdo
    distances_eu["psl_exl_1"] = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]
    distances_eu["psl_exl_2"] = (dist.euclidean((points["x_38"], points["y_38"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]
    distances_mh["psl_exl_1"] = (dist.cityblock((points["x_37"], points["y_37"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]
    distances_mh["psl_exl_2"] = (dist.cityblock((points["x_38"], points["y_38"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano pupila ao canto interno olho esquerdo
    distances_eu["psl_enl_1"] = (dist.euclidean((points["x_37"], points["y_37"]), (points["x_39"], points["y_39"]))) / \
                                reference["w-pixels"]
    distances_eu["psl_enl_2"] = (dist.euclidean((points["x_38"], points["y_38"]), (points["x_39"], points["y_40"]))) / \
                                reference["w-pixels"]
    distances_mh["psl_enl_1"] = (dist.cityblock((points["x_37"], points["y_37"]), (points["x_39"], points["y_39"]))) / \
                                reference["w-pixels"]
    distances_mh["psl_enl_2"] = (dist.cityblock((points["x_38"], points["y_38"]), (points["x_39"], points["y_40"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano palpebra inferior ao canto externo olho esquerdo
    distances_eu["pil_exl_1"] = (dist.euclidean((points["x_41"], points["y_41"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]
    distances_eu["pil_exl_2"] = (dist.euclidean((points["x_40"], points["y_40"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]
    distances_mh["pil_exl_1"] = (dist.cityblock((points["x_41"], points["y_41"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]
    distances_mh["pil_exl_2"] = (dist.cityblock((points["x_40"], points["y_40"]), (points["x_36"], points["y_36"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano pupila ao canto interno olho esquerdo
    distances_eu["pil_enl_1"] = (dist.euclidean((points["x_41"], points["y_41"]), (points["x_39"], points["y_39"]))) / \
                                reference["w-pixels"]
    distances_eu["pil_enl_2"] = (dist.euclidean((points["x_40"], points["y_40"]), (points["x_39"], points["y_39"]))) / \
                                reference["w-pixels"]
    distances_mh["pil_enl_1"] = (dist.cityblock((points["x_41"], points["y_41"]), (points["x_39"], points["y_39"]))) / \
                                reference["w-pixels"]
    distances_mh["pil_enl_2"] = (dist.cityblock((points["x_40"], points["y_40"]), (points["x_39"], points["y_39"]))) / \
                                reference["w-pixels"]

    # Distancia entre os pontos medianos da palpebra superior e inferior do olho direito.
    distances_eu["psr_pir_1"] = (dist.euclidean((points["x_43"], points["y_43"]), (points["x_47"], points["y_47"]))) / \
                                reference["w-pixels"]
    distances_eu["psr_pir_2"] = (dist.euclidean((points["x_44"], points["y_44"]), (points["x_46"], points["y_46"]))) / \
                                reference["w-pixels"]
    distances_mh["psr_pir_1"] = (dist.cityblock((points["x_43"], points["y_43"]), (points["x_47"], points["y_47"]))) / \
                                reference["w-pixels"]
    distances_mh["psr_pir_2"] = (dist.cityblock((points["x_44"], points["y_44"]), (points["x_46"], points["y_46"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano palpebra superior ao canto externo olho esquerdo
    distances_eu["psr_exr_1"] = (dist.euclidean((points["x_43"], points["y_43"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]
    distances_eu["psr_exr_2"] = (dist.euclidean((points["x_44"], points["y_44"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]
    distances_mh["psr_exr_1"] = (dist.cityblock((points["x_43"], points["y_43"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]
    distances_mh["psr_exr_2"] = (dist.cityblock((points["x_44"], points["y_44"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano pupila ao canto interno olho esquerdo
    distances_eu["psr_enr_1"] = (dist.euclidean((points["x_43"], points["y_43"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]
    distances_eu["psr_enr_2"] = (dist.euclidean((points["x_44"], points["y_44"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]
    distances_mh["psr_enr_1"] = (dist.cityblock((points["x_43"], points["y_43"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]
    distances_mh["psr_enr_2"] = (dist.cityblock((points["x_44"], points["y_44"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano palpebra inferior ao canto externo olho esquerdo
    distances_eu["pir_exr_1"] = (dist.euclidean((points["x_47"], points["y_47"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]
    distances_eu["pir_exr_2"] = (dist.euclidean((points["x_46"], points["y_46"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]
    distances_mh["pir_exr_1"] = (dist.cityblock((points["x_47"], points["y_47"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]
    distances_mh["pir_exr_2"] = (dist.cityblock((points["x_46"], points["y_46"]), (points["x_45"], points["y_45"]))) / \
                                reference["w-pixels"]

    # distancia ponto mediano pupila ao canto interno olho esquerdo
    distances_eu["pir_enr_1"] = (dist.euclidean((points["x_47"], points["y_47"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]
    distances_eu["pir_enr_2"] = (dist.euclidean((points["x_46"], points["y_46"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]
    distances_mh["pir_enr_1"] = (dist.cityblock((points["x_47"], points["y_47"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]
    distances_mh["pir_enr_2"] = (dist.cityblock((points["x_46"], points["y_46"]), (points["x_42"], points["y_42"]))) / \
                                reference["w-pixels"]

    # distancia do espaco entre nariz e labio superior
    distances_eu["sn_ls"] = (dist.euclidean((points["x_33"], points["y_33"]), (points["x_51"], points["y_51"]))) / \
                            reference["w-pixels"]
    distances_mh["sn_ls"] = (dist.cityblock((points["x_33"], points["y_33"]), (points["x_51"], points["y_51"]))) / \
                            reference["w-pixels"]
    # comprimento da parte inferior do rosto, nariz ate o queixo
    distances_eu["sn_gn"] = (dist.euclidean((points["x_33"], points["y_33"]), (points["x_8"], points["y_8"]))) / \
                            reference["w-pixels"]
    distances_mh["sn_gn"] = (dist.cityblock((points["x_33"], points["y_33"]), (points["x_8"], points["y_8"]))) / \
                            reference["w-pixels"]
    # largura de ponta a ponta da boca
    distances_eu["chl_chr"] = (dist.euclidean((points["x_48"], points["y_48"]), (points["x_54"], points["y_54"]))) / \
                              reference["w-pixels"]
    distances_mh["chl_chr"] = (dist.cityblock((points["x_48"], points["y_48"]), (points["x_54"], points["y_54"]))) / \
                              reference["w-pixels"]
    # distancia do ponto entre o meio dos olhos e ponto medio da boca
    distances_eu["n_sto"] = (dist.euclidean((points["x_27"], points["y_27"]), (points["x_62"], points["y_62"]))) / \
                            reference["w-pixels"]
    distances_mh["n_sto"] = (dist.cityblock((points["x_27"], points["y_27"]), (points["x_62"], points["y_62"]))) / \
                            reference["w-pixels"]
    # parte inferior do rosto: do ponto medio da boca ate o queixo
    distances_eu["gn_sto"] = (dist.euclidean((points["x_8"], points["y_8"]), (points["x_62"], points["y_62"]))) / \
                             reference["w-pixels"]
    distances_mh["gn_sto"] = (dist.cityblock((points["x_8"], points["y_8"]), (points["x_62"], points["y_62"]))) / \
                             reference["w-pixels"]
    # parte inferior do rosto: comprimento do queixo
    distances_eu["si_gn"] = (dist.euclidean((points["x_57"], points["y_57"]), (points["x_8"], points["y_8"]))) / \
                            reference["w-pixels"]
    distances_mh["si_gn"] = (dist.cityblock((points["x_57"], points["y_57"]), (points["x_8"], points["y_8"]))) / \
                            reference["w-pixels"]
    # altura do perfil inferior FARKAS et al. p 47
    distances_eu["prn_gn"] = (dist.euclidean((points["x_30"], points["y_30"]), (points["x_8"], points["y_8"]))) / \
                             reference["w-pixels"]
    distances_mh["prn_gn"] = (dist.cityblock((points["x_30"], points["y_30"]), (points["x_8"], points["y_8"]))) / \
                             reference["w-pixels"]
    # Metade inferior da altura craniofacial FARKAS et al p 47
    distances_eu["enl_gn"] = (dist.euclidean((points["x_38"], points["y_38"]), (points["x_8"], points["y_8"]))) / \
                             reference["w-pixels"]
    distances_mh["enl_gn"] = (dist.cityblock((points["x_38"], points["y_38"]), (points["x_8"], points["y_8"]))) / \
                             reference["w-pixels"]
    # distancia da glabela subnasal FARKAS et al p 47. Deveria ser 'g_sn' porem nao temos saida do ponto 'g'
    distances_eu["n_sn"] = (dist.euclidean((points["x_27"], points["y_27"]), (points["x_33"], points["y_33"]))) / \
                           reference["w-pixels"]
    distances_mh["n_sn"] = (dist.cityblock((points["x_27"], points["y_27"]), (points["x_33"], points["y_33"]))) / \
                           reference["w-pixels"]
    # distancia do canto interior esquerdo para o nariz
    distances_eu["enl_se"] = (dist.euclidean((points["x_39"], points["y_39"]), (points["x_28"], points["y_28"]))) / \
                             reference["w-pixels"]
    distances_mh["enl_se"] = (dist.cityblock((points["x_39"], points["y_39"]), (points["x_28"], points["y_28"]))) / \
                             reference["w-pixels"]
    # distancia do canto interior esquerdo para o nariz
    distances_eu["enr_se"] = (dist.euclidean((points["x_42"], points["y_42"]), (points["x_28"], points["y_28"]))) / \
                             reference["w-pixels"]
    distances_mh["enr_se"] = (dist.cityblock((points["x_42"], points["y_42"]), (points["x_28"], points["y_28"]))) / \
                             reference["w-pixels"]

    log.info("{0} distances calculated".format(len(distances_eu)))
    return distances_eu, distances_mh


def all(points, reference):
    distances_eu = {}
    distances_mh = {}
    dist_calculated = {}

    for i in range(0, 68):
        x1 = "x_" + str(i)
        y1 = "y_" + str(i)
        for k in range(0, 68):
            key = ""
            aux_key_ik = "p" + str(i) + "_q" + str(k)
            aux_key_ki = "p" + str(k) + "_q" + str(i)

            if i == k:
                continue

            if dist_calculated.get(aux_key_ik, 0) == 0 and dist_calculated.get(aux_key_ki, 0) == 0:
                dist_calculated[aux_key_ik] = 1
                key = "p" + str(i) + "_q" + str(k)

            if key == "" or key != "" and key in distances_eu:
                continue

            x2 = "x_" + str(k)
            y2 = "y_" + str(k)
            distances_eu[key] = (dist.euclidean((points[x1], points[y1]), (points[x2], points[y2]))) / reference[
                "w-pixels"]
            distances_mh[key] = (dist.cityblock((points[x1], points[y1]), (points[x2], points[y2]))) / reference[
                "w-pixels"]

    log.info("{0} distances_eu calculated".format(len(distances_eu)))
    log.info("{0} distances_mh calculated".format(len(distances_mh)))
    return distances_eu, distances_mh