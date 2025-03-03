class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 10572
    metric = 'bias'
    easy_margin = False
    use_se = False
    loss = 'cross_entropy'

    display = False
    finetune = False

    # /home/mathos/Documents/cs/bdrp/repos/arcface-pytorch
    
    train_root = '/kaggle/input/imgs-subset/imgs'
    train_list = '/kaggle/working/ArcFace/lfw_test_pair.txt'
    val_list = '/kaggle/working/ArcFace/lfw_test_pair.txt'

    test_root = '/kaggle/input/dataset2/imgs_subset_2000/test'
    test_list = '/kaggle/working/ArcFace/lfw_test_pair.txt'

    lfw_root = '/kaggle/input/dataset2/imgs_subset_2000'
    lfw_test_list = '/kaggle/working/ArcFace/lfw_test_pair.txt'

    checkpoints_path = '/kaggle/working/ArcFace/checkpoints/'
    # load_model_path = 'models/resnet18.pth'
    # test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 1

    train_batch_size = 64  # batch size
    test_batch_size = 256

    input_shape = (3, 112, 112)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 12  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 0.01  # initial learning rate
    lr_step = 5000
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-3
    momentum=0.9

    bias_model_lambda = 0.01
    num_bias_embedding = 409 # Size of the embedding used for bias prediction
