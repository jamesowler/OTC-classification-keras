params = dict()

params['num_classes'] = 4
params['batch_size'] = 20
params['image_size_X'] = 512
params['image_size_Y'] = 496
params['nb_epoch'] = 100
params['learning_rate'] = 5e-5
params['loss_method'] = 'categorical_crossentropy'
params['augment'] = True
params['data_folder'] = './data'
params['f_name'] = 'OTC-classifier'
params['validation_proportion'] = 0.1
params['validation'] = True
params['verbose'] = True