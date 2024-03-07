import argparse
from gsm8k_evals import db_tools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display leaderboard')
    parser.add_argument('--db',
                        dest='db_name',
                        default='results.db',
                        help='sqlite database for storing results')
    parser.add_argument('--min',
                        dest='min_obs',
                        default=0,
                        type=int,
                        help='minimum obs to show up on leaderboard')
    args = parser.parse_args()
    db_name = args.db_name
    min_obs = args.min_obs
    board = db_tools.leaderboard(db_name,min_obs)
    print(board)
    # this can be easily cleaned up
    fields = ["model_name","sub_set","prompt","struct","sampler", "num_samples",
              "total","maj_acc","pass_acc"]
    field_len = {
        v: max([len(v)]+[len(str(result[i])) for result in board])
        for i, v in enumerate(fields)
    }
    header = [k.ljust(v,' ') for k,v in field_len.items()]
    print(f"<<<{header}>>>")
    header_str = "{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}".format(*header)
    print(header_str)
    print("-"*len(header_str))
    for result in board:
        row_str = "|".join([str(result[i]).ljust(v,' ') 
                   for i,v in enumerate(field_len.values())])
        print(row_str)

