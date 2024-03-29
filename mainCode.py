import search
from datetime import datetime

t = datetime.now()
time = int(t.strftime("%H"))
trafficType = ['heavy', 'mixed', 'light']

def arrivalTime(dist, vel, trafficVol):
    kph = vel * 1.609
    opmltrvltime = dist / kph
    if (time >= 7 and time <= 9) or (time >= 15 and time <= 18):
        if trafficVol == 'heavy':
            return opmltrvltime * 1.5
        elif trafficVol == 'mixed':
            return opmltrvltime * 1.26
        else:
            return opmltrvltime
    else:
        return opmltrvltime


locs = {'SanJuan': (2044.22708, 805.68513), 'Ponce': (1993.03845, 752.62554), 'Salinas': (1989.77668, 786.16885),
        'Adjuntas': (2009.7660, 740.86058), 'Yauco': (1995.37604, 727.61380), 'Maricao': (2010.5420, 717.3230),
        'Rincon': (2028.71519, 684.92270), 'Aguadilla': (2038.46991, 694.96594), 'Camuy': (2041.50870, 726.68274),
        'Utuado': (2021.09065, 743.12228), 'Arecibo': (2040.95274, 748.58215), 'Bayamon': (2035.60690, 799.51964),
        'Catano': (2041.98468, 802.56411), 'Cayey': (2004.87415, 799.92419), 'Caguas': (2019.14630, 813.54432),
        'Juncos': (2017.81584, 191.06579), 'Canovanas': (2034.10634, 193.57450),
        'Humacao': (2009.04663, 200.84064), 'Yabucoa': (1998.14964, 195.16911), 'Guayama': (1990.80829, 805.67548),
        'Fajardo': (2028.27161, 219.65135), 'Mayaguez': (2013.45329, 696.16194)}
PR_map = search.Map({('Ponce', 'Salinas'): arrivalTime(36, 50, trafficType[0]),
                     ('Ponce', 'Adjuntas'): arrivalTime(25, 35, trafficType[2]),
                     ('Ponce', 'Yauco'): arrivalTime(32, 55, trafficType[0]),
                     ('Mayaguez', 'Yauco'): arrivalTime(45, 60, trafficType[0]),
                     ('Mayaguez', 'Maricao'): arrivalTime(26, 30, trafficType[1]),
                     ('Mayaguez', 'Rincon'): arrivalTime(23, 50, trafficType[0]),
                     ('Mayaguez', 'Aguadilla'): arrivalTime(28, 55, trafficType[0]),
                     ('Rincon', 'Aguadilla'): arrivalTime(17, 55, trafficType[0]),
                     ('Camuy', 'Aguadilla'): arrivalTime(39, 45, trafficType[0]),
                     ('Camuy', 'Utuado'): arrivalTime(44, 35, trafficType[2]),
                     ('Camuy', 'Arecibo'): arrivalTime(15, 50, trafficType[0]),
                     ('Utuado', 'Arecibo'): arrivalTime(33, 50, trafficType[1]),
                     ('Utuado', 'Adjuntas'): arrivalTime(21, 30, trafficType[2]),
                     ('Adjuntas', 'Maricao'): arrivalTime(54, 25, trafficType[2]),
                     ('Bayamon', 'Arecibo'): arrivalTime(77, 50, trafficType[0]),
                     ('Bayamon', 'Catano'): arrivalTime(11, 45, trafficType[1]),
                     ('Bayamon', 'SanJuan'): arrivalTime(19, 55, trafficType[0]),
                     ('SanJuan', 'Cayey'): arrivalTime(55, 55, trafficType[0]),
                     ('SanJuan', 'Caguas'): arrivalTime(33, 55, trafficType[0]),
                     ('SanJuan', 'Canovanas'): arrivalTime(30, 40, trafficType[1]),
                     ('Cayey', 'Salinas'): arrivalTime(32, 50, trafficType[1]),
                     ('Cayey', 'Guayama'): arrivalTime(42, 55, trafficType[1]),
                     ('Cayey', 'Caguas'): arrivalTime(24, 55, trafficType[0]),
                     ('Caguas', 'Juncos'): arrivalTime(15, 40, trafficType[1]),
                     ('Canovanas', 'Juncos'): arrivalTime(21, 50, trafficType[1]),
                     ('Canovanas', 'Fajardo'): arrivalTime(31, 50, trafficType[0]),
                     ('Humacao', 'Fajardo'): arrivalTime(36, 55, trafficType[0]),
                     ('Humacao', 'Yabucoa'): arrivalTime(18, 55, trafficType[1]),
                     ('Guayama', 'Salinas'): arrivalTime(23, 50, trafficType[0]),
                     ('Guayama', 'Yabucoa'): arrivalTime(38, 40, trafficType[1])}, locs)