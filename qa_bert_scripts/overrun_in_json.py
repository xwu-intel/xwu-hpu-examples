import json
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_file",
                    default=None,
                    type=str,
                    required=True,
                    help="input file, which json file to overrun")
parser.add_argument("-o", "--output_file",
                    default=None,
                    type=str,
                    required=True,
                    help="output file, the generated new json")
parser.add_argument("--train_bs",
                    default=None,
                    type=int,
                    required=False,
                    help="modified train batch size")
parser.add_argument("--micro_bs",
                    default=None,
                    type=int,
                    required=False,
                    help="modified micro batch size")
parser.add_argument("--steps_per_print",
                    default=None,
                    type=int,
                    required=False,
                    help="modified steps_per_print value")
parser.add_argument("--reduce_bucket_size",
                    default=None,
                    type=int,
                    required=False,
                    help="modified reduce_bucket_size value")
parser.add_argument("--optimizer_type",
                    default=None,
                    type=str,
                    required=False,
                    help="modifies / adds an optimizer to the json file")
parser.add_argument("--overlap_comm",
                    default=None,
                    type=lambda x: (str(x).strip().lower() in ['true','1','yes']),
                    required=False,
                    help="modifies ovelap_comm in zero config")
parser.add_argument("--use_all_reduce",
                    default=None,
                    type=lambda x: (str(x).strip().lower() in ['true','1','yes']),
                    required=False,
                    help="modifies stage3_use_all_reduce_for_fetch_params in zero config")
parser.add_argument("--wall_clock_breakdown",
                    default=None,
                    type=lambda x: (str(x).strip().lower() in ['true', 'false']),
                    required=False,
                    help="modifies wall_clock_breakdown")
parser.add_argument("--prescale_gradients",
                    default=None,
                    type=lambda x: (str(x).strip().lower() in ['true', 'false']),
                    required=False,
                    help="modifies prescale_gradients")
parser.add_argument("--tensorboard_enabled",
                    default=None,
                    type=lambda x: (str(x).strip().lower() in ['true', 'false']),
                    required=False,
                    help="modifies tensorboard_enabled")
args = parser.parse_args()

# Opening JSON file
f = open(args.input_file)
data = json.load(f)

if args.train_bs is not None:
    data['train_batch_size'] = args.train_bs
if args.micro_bs is not None:
    data['train_micro_batch_size_per_gpu'] = args.micro_bs
if args.steps_per_print is not None:
    data['steps_per_print'] = args.steps_per_print
if args.reduce_bucket_size is not None:
    data['zero_optimization']['reduce_bucket_size'] = args.reduce_bucket_size
if args.optimizer_type is not None:
    data['optimizer'] = {'type': args.optimizer_type}
if args.overlap_comm is not None:
    data['zero_optimization']['overlap_comm'] = args.overlap_comm
if args.use_all_reduce is not None:
    data['zero_optimization']['stage3_use_all_reduce_for_fetch_params'] = args.use_all_reduce
if args.wall_clock_breakdown is not None:
    data['wall_clock_breakdown'] = args.wall_clock_breakdown
if args.wall_clock_breakdown is not None:
    data['prescale_gradients'] = args.prescale_gradients
if args.tensorboard_enabled is not None:
    data['tensorboard']['enabled'] = args.tensorboard_enabled

f.close()
json_file = open(args.output_file, "w")
json.dump(data, json_file, indent=2)
json_file.close()
