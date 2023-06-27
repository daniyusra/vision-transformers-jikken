import torch
import torch.nn as nn
from utils import load_pretrained

def load_model(args):
    bs = int(args.bs)
    imsize = int(args.size)

    if args.net=="vit_timm":
        size = 384
    else:
        size = imsize


    if args.net=="vit_small":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="vit_tiny":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="simplevit":
        from models.simplevit import SimpleViT
        net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=="vit":
        # ViT for cifar10
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )


    elif args.net=="vit_timm":
        import timm
        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
   
    elif args.net=="swin":
        from models.swin import swin_t
        net = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))

    elif args.net == "swinpool":
        from models.swin_pool import SwinPool
        if (args.normlayer != ""):
            net = SwinPool(window_size=args.patch, num_classes=10, img_size=size, norm_layer=args.normlayer)
        else:
            net = SwinPool(window_size=args.patch, num_classes=10, img_size=size)
        #net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpool-24":
        from models.swin_pool import SwinPool
        if (args.normlayer != ""):
            net = SwinPool(window_size=args.patch, num_classes=10, img_size=size, norm_layer=args.normlayer, depths = [4,4,12,4])
        else:
            net = SwinPool(window_size=args.patch, num_classes=10, img_size=size, depths = [4,4,12,4])
        #net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpool-24-2":
        from models.swin_pool import SwinPool
        if (args.normlayer != ""):
            net = SwinPool(window_size=args.patch, num_classes=10, img_size=size, norm_layer=args.normlayer, depths = [2,2,18,2])
        else:
            net = SwinPool(window_size=args.patch, num_classes=10, img_size=size, depths = [2,2,18,2])
        #net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpool-exp":
        from models.swin_pool_layerexperiments import SwinPool
        net = SwinPool(window_size=args.patch, num_classes=10, img_size=size)
        

    elif args.net == "swinpoolconv1d":
        from models.swin_poolconv1d import SwinPoolConv1D
        net = SwinPoolConv1D(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())


    elif args.net == "swinpoolconv2d":
        from models.swin_poolconv2d import SwinPoolConv2D
        net = SwinPoolConv2D(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpoolconv2d-i2":
        from models.swin_poolconv2d_batchnorm import SwinPoolConv2DBatchNorm
        net = SwinPoolConv2DBatchNorm(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())
    
    elif args.net == "swinpoolconv2d-nobias":
        from models.swin_poolconv2d_nobias  import SwinPoolConv2DBatchNorm
        net = SwinPoolConv2DBatchNorm(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpool-nobias":
        from models.swin_pool_nobias import SwinPool
        net = SwinPool(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpoolconv2d-i5":
        from models.swin_poolconv2d_i5 import SwinPoolConv2DExperimental
        net = SwinPoolConv2DExperimental(window_size=args.patch, num_classes=10, img_size=size, embed_dim = 96)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpoolconv2d-i3":
        from models.swin_poolconv2d_i3 import SwinPoolConv2DExperimental
        net = SwinPoolConv2DExperimental(window_size=args.patch, num_classes=10, img_size=size, embed_dim = 128)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())
    
    elif args.net == "swinpoolconv2d-i2-newdim":
        from models.swin_poolconv2d_batchnorm import SwinPoolConv2DBatchNorm
        net = SwinPoolConv2DBatchNorm(window_size=args.patch, num_classes=10, img_size=size, embed_dim=64)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinpoolptq":
        from models.swin_pool_ptq import SwinPoolFormersPTQuantizable
        net = SwinPoolFormersPTQuantizable(window_size=args.patch, num_classes=10, img_size=size)
        qnet = SwinPoolFormersPTQuantizable(window_size=args.patch, num_classes=10, img_size=size, q=True)
        quantize = True;

    elif args.net == "swinpool-mlp":
        from models.swin_pool_mlp import SwinPoolMLP
        if (args.normlayer != ""):
            net = SwinPoolMLP(window_size=args.patch, num_classes=10, img_size=size, norm_layer=args.normlayer)
        else:
            net = SwinPoolMLP(window_size=args.patch, num_classes=10, img_size=size)   

    elif args.net == "swinpool-mlp-conv2d":
        from models.swin_pool_mlp_conv2d import SwinPoolMLPConv2D
        if (args.normlayer != ""):
            net = SwinPoolMLPConv2D(window_size=args.patch, num_classes=10, img_size=size, norm_layer=args.normlayer)
        else:
            net = SwinPoolMLPConv2D(window_size=args.patch, num_classes=10, img_size=size)                   
        

    elif args.net == "swinpooltpretrained22":
        from models.swin_pool import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        #net.load_state_dict(torch.load("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth"))
        #net.eval()
        load_pretrained("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth", net)


    elif args.net == "swinoffconv2D":
        from models.swin_official_conv2d import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        #net.load_state_dict(torch.load("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth"))
        #net.eval()

    elif args.net == "swinabl":
        from models.swin_abl import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif args.net == "swinabl-s":
        from models.swin_abl import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size, depths=[2, 2, 18, 2])
        #window_size =args.patch, num_classes=10, downscaling_factors=(2,2,2,1))

    elif args.net == "swinoff":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        print(net.flops())

    elif args.net == "swinoff-t-22pretrained":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size)
        for param in net.parameters():
            print(type(param), param.size())
        #net.load_state_dict(torch.load("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth"))
        #net.eval()
        load_pretrained("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth", net)

    elif args.net == "swinoff-s-22pretrained":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size, depths=[2,2,18,2])
        #net.load_state_dict(torch.load("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth"))
        #net.eval()
        load_pretrained("./pretrainedmodels/swin_small_patch4_window7_224_22k.pth", net)

    elif args.net =="swinoff-b":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=args.patch, num_classes=10, img_size=size, depths = [2,2,18,2], embed_dim = 128, num_heads = (4,8,16,32))
    

    elif args.net == "poolformer":
        from models.vit_pool import PoolFormer
        net = PoolFormer(layers=[2, 2, 6, 2], embed_dims=[64, 128, 320, 512], mlp_ratios= [4, 4, 4, 4], downsamples =[True, True, True, True]);

    elif args.net == "poolformer_linear":
        from models.vit_pool_linear import PoolFormer
        net = PoolFormer(layers=[2, 2, 6, 2], embed_dims=[64, 128, 320, 512], mlp_ratios= [4, 4, 4, 4], downsamples =[True, True, True, True]);
    return net
