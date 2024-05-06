import pandas as pd
import numpy as np
#pd.set_option("display.max_rows", None, "display.max_columns", None)


if __name__=="__main__":
    full_data_path = 'data_5fps/val/8_tra_new_val.csv'
    data = pd.read_csv(full_data_path, sep=',', index_col=False, header=0)  # change delimiter seperator
    data.sort_values('recordingId', inplace=True)
    count = 0
    mode = "val"
    l_c_c, t_c_c, r_c_c, b_c_c =0, 0, 0, 0
    l_u_c, t_u_c, r_u_c, b_u_c =0, 0, 0, 0
    l_n_c, t_n_c, r_n_c, b_n_c =0, 0, 0, 0
    for case_id in pd.unique(data['recordingId']):
        print("processing recordingId: ", case_id)
        l_c, t_c, r_c, b_c = [], [], [], []  # clump region at different junctions
        l_u, t_u, r_u, b_u = [], [], [], []  # unclump region at different junctions
        l_n, t_n, r_n, b_n = [], [], [], []  # neutral region at different junctions
        d = data[ data['recordingId'] == case_id ]

        d = d[ d['class'] == 'car']
        d.sort_values('frame', inplace=True)
        d = d[ ["frame", "trackId", "xCenter", "yCenter", "cluster"] ] # filter only required columns
        if len(d) == 0:
            continue

        #clump check
        l_c = d[ (d['xCenter'] >= 18) & (d['xCenter'] <= 33) & (d['yCenter'] >= -40) & (d['yCenter'] <= -32)] # left junction
        t_c = d[ (d['xCenter'] >= 85) & (d['xCenter'] <= 97) & (d['yCenter'] >= -19) & (d['yCenter'] <= -12) ] # top   junction
        r_c = d[ (d['xCenter'] >= 123) & (d['xCenter'] <= 138) & (d['yCenter'] >= -62) & (d['yCenter'] <= -53) ] # right   junction
        b_c = d[ (d['xCenter'] >= 68) & (d['xCenter'] <= 79) & (d['yCenter'] >= -79) & (d['yCenter'] <= -73) ] # bottom   junction
        # unclump check
        l_u = d[ (d['xCenter'] >= 48) & (d['xCenter'] <= 62) & (d['yCenter'] >= -53) & (d['yCenter'] <= -40)] # left junction
        t_u = d[ (d['xCenter'] >= 72) & (d['xCenter'] <= 88) & (d['yCenter'] >= -32) & (d['yCenter'] <= -22) ] # top   junction
        r_u = d[ (d['xCenter'] >= 95) & (d['xCenter'] <= 110) & (d['yCenter'] >= -52) & (d['yCenter'] <= -40)] # right   junction
        b_u = d[ (d['xCenter'] >= 75) & (d['xCenter'] <= 90) & (d['yCenter'] >= -72) & (d['yCenter'] <= -66) ] # bottom   junction
        # neutral check
        l_n = d[ (d['xCenter'] >= 32) & (d['xCenter'] <= 48) & (d['yCenter'] >= -48) & (d['yCenter'] <= -36) ] # left junction
        t_n = d[ (d['xCenter'] >= 80) & (d['xCenter'] <= 93) & (d['yCenter'] >= -24) & (d['yCenter'] <= -18) ] # top   junction
        r_n = d[ (d['xCenter'] >= 108) & (d['xCenter'] <= 123) & (d['yCenter'] >= -58) & (d['yCenter'] <= -43) ] # right   junction
        b_n = d[ (d['xCenter'] >= 70) & (d['xCenter'] <= 80) & (d['yCenter'] >= -73) & (d['yCenter'] <= -70) ] # bottom   junction

        # clump
        # get boolean values where each object has at least 20 positions
        l_c_check = np.array([ len(pd.unique(l_c[ l_c['trackId'] == id]['frame']))>=20 for id in pd.unique(l_c['trackId'])])
        t_c_check = np.array([len(pd.unique(t_c[t_c['trackId'] == id]['frame'])) >= 20 for id in pd.unique(t_c['trackId'])])
        r_c_check = np.array([len(pd.unique(r_c[r_c['trackId'] == id]['frame'])) >= 20 for id in pd.unique(r_c['trackId'])])
        b_c_check = np.array([len(pd.unique(b_c[b_c['trackId'] == id]['frame'])) >= 20 for id in pd.unique(b_c['trackId'])])

        if  len(l_c) > 0 and len(pd.unique(l_c['trackId'])) > 1 and len(np.where(l_c_check == True)[0]) >=2:
            l_c.to_csv('../rounD/state_5fps/{}/rounD_left_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            l_c_c+=1
        if  len(t_c) > 0 and len(pd.unique(t_c['trackId'])) > 1 and len(np.where(t_c_check == True)[0]) >=2:
            t_c.to_csv('../rounD/state_5fps/{}/rounD_top_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            t_c_c+=1
        if  len(r_c) > 0 and len(pd.unique(r_c['trackId'])) > 1 and len(np.where(r_c_check == True)[0]) >=2:
            r_c.to_csv('../rounD/state_5fps/{}/rounD_right_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            r_c_c+=1
        if  len(b_c) > 0 and len(pd.unique(b_c['trackId'])) > 1 and len(np.where(b_c_check == True)[0]) >=2:
            b_c.to_csv('../rounD/state_5fps/{}/rounD_bottom_clump_{}.txt'.format(mode, str(case_id)),header=None,index=None, sep=',')
            b_c_c+=1

        # unclump
        # get boolean values where each object has atleast 20 positions
        l_u_check = np.array([len(pd.unique(l_u[l_u['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(l_u['trackId'])])
        t_u_check = np.array([len(pd.unique(t_u[t_u['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(t_u['trackId'])])
        r_u_check = np.array([len(pd.unique(r_u[r_u['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(r_u['trackId'])])
        b_u_check = np.array([len(pd.unique(b_u[b_u['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(b_u['trackId'])])

        if len(l_u) > 0 and len(pd.unique(l_u['trackId'])) > 1 and len(np.where(l_u_check == True)[0]) >=2:
            l_u.to_csv('../rounD/state_5fps/{}/rounD_left_unclump_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_u_c+=1
        if len(t_u) > 0 and len(pd.unique(t_u['trackId'])) > 1 and len(np.where(t_u_check == True)[0]) >=2:
            t_u.to_csv('../rounD/state_5fps/{}/rounD_top_unclump_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            t_u_c+=1
        if len(r_u) > 0 and len(pd.unique(r_u['trackId'])) > 1 and len(np.where(r_u_check == True)[0]) >=2:
            r_u.to_csv(
                '../rounD/state_5fps/{}/rounD_right_unclump_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            r_u_c+=1
        if len(b_u) > 0 and len(pd.unique(b_u['trackId'])) > 1 and len(np.where(b_u_check == True)[0]) >=2:
            b_u.to_csv(
                '../rounD/state_5fps/{}/rounD_bottom_unclump_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            b_u_c+=1

        # neutral
        # get boolean values where each object has atleast 20 positions
        l_n_check = np.array([len(pd.unique(l_n[l_n['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(l_n['trackId'])])
        t_n_check = np.array([len(pd.unique(t_n[t_n['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(t_n['trackId'])])
        r_n_check = np.array([len(pd.unique(r_n[r_n['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(r_n['trackId'])])
        b_n_check = np.array([len(pd.unique(b_n[b_n['trackId'] == id]['frame'])) >= 20 for id in
                     pd.unique(b_n['trackId'])])
        if len(l_n) > 0 and len(pd.unique(l_n['trackId'])) > 1 and len(np.where(l_n_check == True)[0]) >=2:
            l_n.to_csv('../rounD/state_5fps/{}/rounD__left_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            l_n_c+=1
        if len(t_n) > 0 and len(pd.unique(t_n['trackId'])) > 1 and len(np.where(t_n_check == True)[0]) >=2:
            t_n.to_csv('../rounD/state_5fps/{}/rounD__top_neutral_{}.txt'.format(mode, str(case_id)),
                       header=None, index=None, sep=',')
            t_n_c+=1
        if len(r_n) > 0 and len(pd.unique(r_n['trackId'])) > 1 and len(np.where(r_n_check == True)[0]) >=2:
            r_n.to_csv(
                '../rounD/state_5fps/{}/rounD__right_neutral_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            r_n_c+=1
        if len(b_n) > 0 and len(pd.unique(b_n['trackId'])) > 1 and len(np.where(b_n_check == True)[0]) >=2:
            b_n.to_csv(
                '../rounD/state_5fps/{}/rounD__neutral_{}.txt'.format(mode, str(case_id)),
                header=None, index=None, sep=',')
            b_n_c+=1

    print("l, t, r, b clump:",l_c_c, t_c_c, r_c_c, b_c_c )
    print("l, t, r, b unclump:", l_u_c, t_u_c, r_u_c, b_u_c)
    print("l, t, r, b neutral:", l_n_c, t_n_c, r_n_c, b_n_c)
    print("clump:",l_c_c+ t_c_c+ r_c_c+ b_c_c )
    print("unclump:",l_u_c+ t_u_c+ r_u_c+ b_u_c )
    print("neutral:",l_n_c+ t_n_c+ r_n_c+ b_n_c )

'''
train data files count:
l, t, r, b clump: 205 99 521 51
l, t, r, b unclump: 1024 304 1 814
l, t, r, b neutral: 3224 1709 647 962
clump: 876
unclump: 2143
neutral: 6542
'''