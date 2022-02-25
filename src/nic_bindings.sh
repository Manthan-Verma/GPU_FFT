#!/bin/bash
export UCX_RNDV_SCHEME=put_zcopy
case $(( ${OMPI_COMM_WORLD_LOCAL_RANK} )) in
    0) UCX_NET_DEVICES=mlx5_0:1 OMPI_MCA_btl_openib_if_include=mlx5_0 CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=3 $* ;;
    1) UCX_NET_DEVICES=mlx5_1:1 OMPI_MCA_btl_openib_if_include=mlx5_1 CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=3 $* ;;
    2) UCX_NET_DEVICES=mlx5_2:1 OMPI_MCA_btl_openib_if_include=mlx5_2 CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=1 $* ;;
    3) UCX_NET_DEVICES=mlx5_3:1 OMPI_MCA_btl_openib_if_include=mlx5_3 CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=1 $* ;;
    4) UCX_NET_DEVICES=mlx5_4:1 OMPI_MCA_btl_openib_if_include=mlx5_4 CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=7 $* ;;
    5) UCX_NET_DEVICES=mlx5_5:1 OMPI_MCA_btl_openib_if_include=mlx5_5 CUDA_VISIBLE_DEVICES=5 numactl --cpunodebind=7 $* ;;
    6) UCX_NET_DEVICES=mlx5_6:1 OMPI_MCA_btl_openib_if_include=mlx5_6 CUDA_VISIBLE_DEVICES=6 numactl --cpunodebind=5 $* ;;
    7) UCX_NET_DEVICES=mlx5_7:1 OMPI_MCA_btl_openib_if_include=mlx5_7 CUDA_VISIBLE_DEVICES=7 numactl --cpunodebind=5 $* ;;
esac