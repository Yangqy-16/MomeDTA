"""
Modified from https://github.com/westlake-repl/SaProt
NOTE: batch_size must be 1!
"""
from saprot.base import SaprotBaseModel
from transformers import EsmTokenizer
from utils import *


if __name__ == "__main__":
    args = parse_arguments()
    root = args.root
    # batch_size = args.bs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    output_dir = f'{root}/embed/saprot'
    os.makedirs(output_dir, exist_ok=True)

    saprot_root = "/data/qingyuyang/dta_ours/weights/saprot"

    # model = EsmModel.from_pretrained(saprot_root)
    # tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(saprot_root)
    config = {
        "task": "base",
        "config_path": saprot_root, # Note this is the directory path of SaProt, not the ".pt" file
        "load_pretrained": True,
    }
    model = SaprotBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    print("Finish loading model.")

    # MomeDTA
    cs_df = pd.read_csv(f'{root}/prots.csv')
    cs_list = list(set(zip(cs_df['prot_id'], cs_df['sa_seq'])))

    cs_list = select_undone_items(cs_list, output_dir)

    for idx, sa_seq in tqdm(cs_list):
        # seqs_batch = cs_list[idx : min(idx + batch_size, len(cs_list))]
        # embeddings = infer_batch([x[1] for x in seqs_batch])
        inputs = tokenizer(sa_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embed = model.get_hidden_states(inputs)#.last_hidden_state

        torch.save(embed[0].clone().detach().cpu(), f'{output_dir}/{idx}.pt')
