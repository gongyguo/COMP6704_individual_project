python -u main.py --method node2vec --dataset cora|tee -a log_cora
python -u main.py --method deepwalk --dataset cora|tee -a log_cora
python -u main.py --method grarep --dataset cora|tee -a log_cora
python -u main.py --method netmf --dataset cora|tee -a log_cora
python -u main.py --method gcn --dataset cora|tee -a log_cora
python -u main.py --method gat --dataset cora|tee -a log_cora
python -u main.py --method gin --dataset cora|tee -a log_cora   

python -u main.py --method node2vec --dataset citeseer|tee -a log_citeseer
python -u main.py --method deepwalk --dataset citeseer|tee -a log_citeseer
python -u main.py --method grarep --dataset citeseer|tee -a log_citeseer
python -u main.py --method netmf --dataset citeseer|tee -a log_citeseer
python -u main.py --method gcn --dataset citeseer|tee -a log_citeseer  
python -u main.py --method gat --dataset citeseer|tee -a log_citeseer
python -u main.py --method gin --dataset citeseer|tee -a log_citeseer

python -u main.py --method node2vec --dataset blogcatalog|tee -a log_balogcatalog
python -u main.py --method deepwalk --dataset blogcatalog|tee -a log_balogcatalog
python -u main.py --method grarep --dataset blogcatalog|tee -a log_balogcatalog
python -u main.py --method netmf --dataset blogcatalog|tee -a log_balogcatalog
python -u main.py --method gcn --dataset blogcatalog|tee -a log_balogcatalog
python -u main.py --method gat --dataset blogcatalog|tee -a log_balogcatalog
python -u main.py --method gin --dataset blogcatalog|tee -a log_balogcatalog

python -u main.py --method node2vec --dataset wiki|tee -a log_wiki
python -u main.py --method deepwalk --dataset wiki|tee -a log_wiki
python -u main.py --method grarep --dataset wiki|tee -a log_wiki   
python -u main.py --method netmf --dataset wiki|tee -a log_wiki
python -u main.py --method gcn --dataset wiki|tee -a log_wiki  
python -u main.py --method gat --dataset wiki|tee -a log_wiki
python -u main.py --method gin --dataset wiki|tee -a log_wiki
