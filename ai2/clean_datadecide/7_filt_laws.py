from ladder.fitting.step1_flops import \
    fit_step1 as fit_step1_flops, \
    predict_step1 as predict_step1_flops, \
    plot_step1 as plot_step1_flops

def add_ladder_data_cheap_decisions(data_by_name):
    """ From Ian """
    sequence_length = 2048

    def model_and_step_to_tokens(model, step):
        return MODEL_TO_BATCH[model] * step * sequence_length

    def model_and_step_to_compute(model, step):
        return MODEL_TO_PARAMETERS[model] * model_and_step_to_tokens(model, step) * 6
    
    for k, v in data_by_name.items():
        step = v['step'][-1]
        c = model_and_step_to_compute(k, step)
        n = MODEL_TO_PARAMETERS[k]
        d = model_and_step_to_tokens(k, step)
        f = float(n * d * 6)
        data_by_name[k]['ns'] = [n]
        data_by_name[k]['fs'] = [f]
        data_by_name[k]["ds"] = [d]

    # raise RuntimeError(data_by_name)

    return data_by_name


MODEL_TO_BATCH = {
    '4M': 32, # batch_size=32, gpus=8
    '6M': 32,
    '8M': 32,
    '10M': 32,
    '14M': 32,
    '16M': 32,
    '20M': 64,
    '60M': 96,
    '90M': 160,
    '150M': 192,
    '300M': 320,
    '530M': 448,
    '750M': 576,
    '1B': 704
}

MODEL_TO_PARAMETERS = {
    '4M': 3_744_832,
    '6M': 6_010_464,
    '8M': 8_538_240,
    '10M': 9_900_432,
    '12M': 12_066_600,
    '14M': 14_380_224,
    '16M': 16_004_560,
    '20M': 19_101_888,
    '60M': 57_078_144,
    '90M': 97_946_640,
    '150M': 151898880,
    '300M': 319980544,
    '530M': 530074944,
    '750M': 681297408,
    '1B': 1_176_832_000
}