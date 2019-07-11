params = dict()

params['num_classes'] = 4
params['batch_size'] = 20
params['image_size_X'] = 512
params['image_size_Y'] = 496
params['nb_epoch'] = 100
params['learning_rate'] = 1e-5
params['loss_method'] = 'categorical_crossentropy'
params['model_name'] = 'OTC_classifier'
params['nb_GPUs'] = 1