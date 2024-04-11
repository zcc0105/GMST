import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser(description="CBIM-BO")

    parser.add_argument('--seeds_size', type=int, default=50,
                        help="the number of seeds attributed to the platform")
    parser.add_argument('--dataset_name', default='dblp', choices=['cit', 'p2p', 'dblp', 'facebook'],
                        help='dataset to quality the influence spread that a competitive order can generate')
    parser.add_argument('--iterations', default=200, choices=[100, 200, 300, 400])
    parser.add_argument('--isPool', action='store_true', default=True)     # 默认是False
    parser.add_argument('--seq_length', type=int, default=10)
    parser.add_argument('--is_Random', action='store_true')
    parser.add_argument('--is_HG', action='store_true')
    parser.add_argument('--is_Greedy', action='store_true')
    parser.add_argument('--is_RG', action='store_true')
    parser.add_argument('--is_GMST', action='store_true')
    parser.add_argument('--is_PRank', action='store_true')
    parser.add_argument('--is_Close', action='store_true')
    parser.add_argument('--is_Between', action='store_false')

    opts = parser.parse_args(args)

    return opts