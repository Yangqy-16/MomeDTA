""" NOTE: batch_size must be 1! """
from utils import *
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig


if __name__ == "__main__":
    args = parse_arguments()
    root = args.root
    # batch_size = args.bs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    output_dir = f'{root}/embed/selformer'
    os.makedirs(output_dir, exist_ok=True)

    model_path = "/data/qingyuyang/dta_ours/weights/selformer" # path of the pre-trained model
    config = RobertaConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path, config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    print("Finish loading model")

    df = pd.read_csv(f'{root}/drugs.csv')
    drug_info = list(set(zip(df['drug_id'], df['selfies'])))
    drug_info = select_undone_items(drug_info, output_dir)

    for idx, selfies in tqdm(drug_info):
    # def get_sequence_embeddings(selfies):
        token = torch.tensor([tokenizer.encode(selfies, max_length=512, truncation=True)]).cuda()#add_special_tokens=True, padding=True, 
        with torch.no_grad():
            output = model(token)

        sequence_out = output[0]
        torch.save(sequence_out[0].clone().detach().cpu(), f'{output_dir}/{idx}.pt')

    # selformer_root = "/data/qingyuyang/dta_ours/weights/selformer"
    # model = AutoModel.from_pretrained(selformer_root)
    # tokenizer = AutoTokenizer.from_pretrained(selformer_root)

        # seqs_batch = selfies[idx : min(idx + batch_size, len(selfies))]
        # embeddings = infer_batch([x[1] for x in seqs_batch])
        # inputs = tokenizer(selfies, return_tensors="pt", max_length=512)  #s, return_tensors="pt", padding=True, truncation=True,  NOTE: Mol len CAN'T be longer than 514 for SELFormer!
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        # with torch.no_grad():
        #     embed = model(**inputs).last_hidden_state
        # # for seq_info, embedding in zip(seqs_batch, embeddings):
        # torch.save(embed[0].clone().detach().cpu(), f'{output_dir}/{idx}.pt')
        # return outputs

    # def generate_selformer_embeddings(selfies: list[tuple[str, str]], output_dir: Path) -> None:

    # generate_selformer_embeddings(selfies, output_dir)
