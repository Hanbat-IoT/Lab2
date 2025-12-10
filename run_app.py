import logging
import server
import os
from options import args_parser

def server_init(params):
    args = args_parser()
    req_rounds = int(params['rounds'])
    req_clients = int(params['clients'])
    req_dataset = params['dataset']
    req_IID = int(params['data_distribution'])
    
    args.rounds = req_rounds
    args.num_clients = req_clients
    args.dataset = req_dataset
    args.IID = req_IID
    

    file_logger = logging.getLogger("File")
    file_logger.setLevel(logging.DEBUG)
    formatter1 = logging.Formatter('%(message)s')
    log_dir = "logs"

    log_dir = log_dir  + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_f_name = log_dir + '[' + str(args.rounds) + "]rounds_" + "[" + \
        str(args.num_clients) + "]clients_[" + str(args.dataset) + "]_[" + str(args.IID) + "]IID.log"
    file_handler = logging.FileHandler(log_f_name)
    
    file_handler.setFormatter(formatter1)
    file_logger.addHandler(file_handler)

    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
    fl_server = server.Server(args, file_logger)
    return fl_server, log_f_name

def server_boot(fl_server):
    fl_server.boot()

def server_run(fl_server, r):
    fl_server.run(r)