"""
This script is from ESM's repository (https://github.com/facebookresearch/esm).

We support batch processing in this script, but note that long sequences may take lots of memory when inferring.
Please set 'batch_size' suitable for your device.

After running this script, you can load the generated ESM-2 representations of proteins by: 
    torch.load('###.pt')
"""

from utils import *


if __name__ == "__main__":
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    batch_size = args.bs

    root = args.root
    save_path = f'{root}/embed/esm2'  # NOTE: can change
    os.makedirs(save_path, exist_ok=True)

    df = pd.read_csv(f'{root}/prots.csv')
    all_list = list(set(zip(df['prot_id'], df['prot_seq'])))
    print('Finish loading protein data')

    all_list = select_undone_items(all_list, save_path)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # can change models of different size; the model is stored in ~/.cache/torch/hub/checkpoints after the 1st time
    model.eval()  # disables dropout for deterministic results
    model.to('cuda')
    print('Finish loading model')

    batch_converter = alphabet.get_batch_converter()

    def infer_batch(data: list[tuple[str, str]]):
        batch_labels, _, batch_tokens = batch_converter(data)  # all proteins' names, all proteins sequences, torch.Size([#proteins, max_protein_length_in_this_batch])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # all sequences' length
        batch_tokens = batch_tokens.to('cuda')

        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations: torch.Tensor = results["representations"][33]  # NOTE: layer number same as model name; less than 1s

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):  #contact in results['contacts']
            label = batch_labels[i]
            output_file = os.path.join(save_path, f"{label}.pt")
            save_contents = token_representations[i, 1 : tokens_len - 1].clone().detach().cpu()
            torch.save(save_contents, output_file)  # torch: [seq_len, embedding_dim]

    for idx in tqdm(range(0, len(all_list), batch_size)):
        data = all_list[idx : min(idx + batch_size, len(all_list))]
        infer_batch(data)
        # break
