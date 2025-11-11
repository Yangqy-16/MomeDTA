""" NOTE: batch_size must be 1! """
from torch import Tensor
import sys
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设preprocess的父目录是根目录）
project_root = os.path.dirname(current_dir)
# 添加到系统路径
sys.path.append(project_root)

from src.model.unimol.unimol import UniMolModel
from src.utils.padding import batch_collate_fn
from utils import *


if __name__ == "__main__":
    args = parse_arguments()
    root = args.root
    batch_size = args.bs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    output_dir = f'{root}/embed/unimol'
    os.makedirs(output_dir, exist_ok=True)

    unimol = UniMolModel('/data/qingyuyang/dta_ours/weights/unimol')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unimol.eval()
    unimol.to(device)
    print("Finish loading model")

    input_dir = f'{root}/token/unimol'
    mol_list = []
    for i in os.listdir(input_dir):
        mol_name = i.split('.')[0]
        mol_conf = torch.load(f'{input_dir}/{i}')
        mol_list.append((mol_name, mol_conf))

    def infer_batch(seqs: list[dict[str, Tensor]]) -> list[torch.Tensor]:
        """
        """
        inputs = batch_collate_fn(seqs)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = unimol(**inputs)['atomic_reprs']  # [seq_len, embed_dim]
        return outputs

    def generate_unimol_embeddings(cs_list: list[tuple[str, str]], output_dir: Path) -> None:
        for idx in tqdm(range(0, len(cs_list), batch_size)):
            seqs_batch = cs_list[idx : min(idx + batch_size, len(cs_list))]
            embeddings = infer_batch([x[1] for x in seqs_batch])
            for seq_info, embedding in zip(seqs_batch, embeddings):
                torch.save(embedding.clone().detach().cpu(), f'{output_dir}/{seq_info[0]}.pt')

    mol_list = select_undone_items(mol_list, output_dir)
    generate_unimol_embeddings(mol_list, output_dir)
