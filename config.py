N_SEGMENTS = 50
GAZE = 'gaze'
GAZE_HP = 'gaze_hp_'
GAZE_HP_AU = 'engage_gaze+hp+au_10'
ALL = 'all_'
GAZE_HP_AU_LMK = 'engage_gaze+hp+au+lmk'
GAZE_HP_AU_LMK_PDM = 'engage_gaze+hp+au+lmk+pdm'
MARLIN = 'marlin_features_large'
BODYPOSE = 'engage_openface_bodypose'
XCLIP = 'engage_xclip'
FUSION = 'engage_gaze+hp+au_marlin'
LABEL_MAP = {
    'Not-Engaged': 0,
    'Barely-engaged': 1,
    'Engaged': 2,
    'Highly-Engaged': 3
}
SNP = 'SNP(Subject Not Present)'