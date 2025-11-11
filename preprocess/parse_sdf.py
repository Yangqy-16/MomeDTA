from rdkit import Chem
import numpy as np


def parse_sdf_file(sdf_path):
    """
    解析SDF文件, 返回原子列表和坐标数组
    
    参数:
        sdf_path (str): SDF文件路径
        
    返回:
        tuple: (atoms, coords)
            - atoms: 原子符号列表，如 ['C', 'O', 'H', ...]
            - coords: 坐标的numpy数组, 形状为(n, 3)
    """
    suppl = Chem.SDMolSupplier(sdf_path)
    mol = next(suppl)  # 获取第一个分子
    
    if mol is None:
        raise ValueError("无法解析SDF文件或文件为空")
    
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
    
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    
    return atoms, np.array(coords)


def parse_sdf_text(sdf_path):
    """
    直接通过文本解析SDF文件, 返回原子列表和坐标数组
    
    参数:
        sdf_path (str): SDF文件路径
        
    返回:
        tuple: (atoms, coords)
            - atoms: 原子符号列表，如 ['C', 'O', ...]
            - coords: 坐标的numpy数组, 形状为(n, 3)
    """
    with open(sdf_path, 'r') as f:
        lines = f.readlines()
    
    # 找到原子数和坐标块的起始行
    counts_line = None
    for i, line in enumerate(lines):
        if line.strip().endswith("V2000"):  # 通常原子数在"V2000"行
            counts_line = lines[i]
            break
    
    if not counts_line:
        raise ValueError("无效的SDF文件: 未找到原子信息块")
    
    # 解析原子数（假设V2000行的前3个数字是原子数）
    try:
        num_atoms = int(counts_line.split()[0])
    except:
        raise ValueError("无法解析原子数")
    
    # 提取原子和坐标
    atoms = []
    coords = []
    start_idx = i + 1  # 原子数据从V2000下一行开始
    end_idx = start_idx + num_atoms
    
    for line in lines[start_idx:end_idx]:
        parts = line.split()
        if len(parts) < 4:
            continue  # 跳过不完整的行
        x, y, z = map(float, parts[:3])
        atom = parts[3]

        if atom == 'H':  # 去除H
            continue
        
        atoms.append(atom)
        coords.append([x, y, z])
    
    return atoms, np.array(coords)


if __name__ == "__main__":
    sdf_file = "/data/yueteng/DTI/PDBbind/Raw/PDBbind_v2020_refined/refined-set/1a1e/1a1e_ligand.sdf"  # 替换为你的SDF文件路径
    atoms, coords = parse_sdf_text(sdf_file)
    
    print("Atoms:", atoms)
    print("Coordinates:\n", coords)
