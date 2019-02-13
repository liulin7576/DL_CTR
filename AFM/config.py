TRAIN_FILE = "../data/temp.pkl"
# TEST_FILE = "data/test.csv"

SUB_DIR = "./output"


NUM_SPLITS = 3
RANDOM_SEED = 2019

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]

NUMERIC_COLS = [
    
# 'adid_model_nuq_num', 'model_adid_nuq_num', 'adid_make_nuq_num', 'make_adid_nuq_num', 'adid_os_nuq_num', 'os_adid_nuq_num', 'adid_city_nuq_num', 'city_adid_nuq_num', 'adid_province_nuq_num', 'province_adid_nuq_num', 'adid_f_channel_nuq_num', 'f_channel_adid_nuq_num', 'adid_app_id_nuq_num', 'app_id_adid_nuq_num', 'adid_carrier_nuq_num', 'carrier_adid_nuq_num', 'adid_nnt_nuq_num', 'nnt_adid_nuq_num', 'adid_devtype_nuq_num', 'devtype_adid_nuq_num', 'adid_app_cate_id_nuq_num', 'app_cate_id_adid_nuq_num', 'adid_inner_slot_id_nuq_num', 'inner_slot_id_adid_nuq_num', 'hour', 'advert_id_rate', 'advert_industry_inner_rate', 'advert_name_rate', 'campaign_id_rate', 'creative_height_rate', 'creative_tp_dnf_rate', 'creative_width_rate', 'province_rate', 'f_channel_rate'
    
]


IGNORE_COLS = [
    
    'advert_id', 'app_paid', 'click', 'creative_is_js', 'creative_is_voicead', 'instance_id', 'make', 'model', 'os_name', 'osv',
    'time', 'user_tags', 'meiti_con_inner', 'meiti_con_inner_con_channel', 'app_cate_con_adid', 'app_cate_con_meiti',
    'model_con_osv', 'model_con_city', 'day', 'period'
    
] + [
    
'adid_model_nuq_num', 'model_adid_nuq_num', 'adid_make_nuq_num', 'make_adid_nuq_num', 'adid_os_nuq_num', 'os_adid_nuq_num', 'adid_city_nuq_num', 'city_adid_nuq_num', 'adid_province_nuq_num', 'province_adid_nuq_num', 'adid_f_channel_nuq_num', 'f_channel_adid_nuq_num', 'adid_app_id_nuq_num', 'app_id_adid_nuq_num', 'adid_carrier_nuq_num', 'carrier_adid_nuq_num', 'adid_nnt_nuq_num', 'nnt_adid_nuq_num', 'adid_devtype_nuq_num', 'devtype_adid_nuq_num', 'adid_app_cate_id_nuq_num', 'app_cate_id_adid_nuq_num', 'adid_inner_slot_id_nuq_num', 'inner_slot_id_adid_nuq_num', 'hour', 'advert_id_rate', 'advert_industry_inner_rate', 'advert_name_rate', 'campaign_id_rate', 'creative_height_rate', 'creative_tp_dnf_rate', 'creative_width_rate', 'province_rate', 'f_channel_rate'
]
    