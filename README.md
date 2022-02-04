# 並列コンピューティング 演習, レポートソースコード
## 実験環境
```
polarisサーバー
Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz
MemTotal:       65868424 kB
論理プロセッサ数: 24

apc2102@capella:~/apc21/mpi$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 20.04.3 LTS
Release:        20.04
Codename:       focal
```

### open mp
```
apc2102@capella:~/apc21/mpi$ g++ --version
g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### mpi
```
apc2102@capella:~/apc21$ mpic++ --version
g++ (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### cuda
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

```
$ nvidia-smi
apc2102@capella:~$ nvidia-smi
Mon Jan 24 19:42:00 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro K4200        Off  | 00000000:03:00.0 Off |                  N/A |
| 34%   58C    P8    22W / 110W |     19MiB /  4035MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A   1004937      G   /usr/lib/xorg/Xorg                 16MiB |
+-----------------------------------------------------------------------------+
```

- `/openmp`: OpenMP演習課題, 行列積
- `/mpi`: MPI演習課題, 行列積
- `/cuda`: CUDA演習課題, 行列積
- `/exp`: 図作成用コード