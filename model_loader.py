import torch
from models.convmixer import ConvMixer
from models.resnet import ResNet101, ResNet18, ResNet34, ResNet50
from models.vgg import VGG
import torch.nn as nn
from utils import load_pretrained

def load_model(netname, imsize, patch, dimhead, num_classes, cuda):
    # Data
    if netname=="vit_timm":
        size = 384
    else:
        size = imsize

    # Model factory..
    print('==> Building model..')
    # net = VGG('VGG19')
    if netname=='res18':
        net = ResNet18()
    elif netname=='vgg':
        net = VGG('VGG19')
    elif netname=='res34':
        net = ResNet34()
    elif netname=='res50':
        net = ResNet50()
    elif netname=='res101':
        net = ResNet101()
    elif netname=="convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=num_classes)
    elif netname=="mlpmixer":
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = patch,
        dim = 512,
        depth = 6,
        num_classes = num_classes
        )
    elif netname=="vit_small":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
        )
    elif netname=="vit_tiny":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
        )
    elif netname=="simplevit":
        from models.simplevit import SimpleViT
        net = SimpleViT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
        )
    elif netname=="vit":
        # ViT for cifar10
        net = ViT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
        )


    elif netname=="vit_timm":
        import timm
        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, num_classes)
    elif netname=="cait":
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
        )

    elif netname=="cait_small":
        from models.cait import CaiT
        net = CaiT(
        image_size = size,
        patch_size = patch,
        num_classes = num_classes,
        dim = int(dimhead),
        depth = 6,   # depth of transformer for patch to patch attention only
        cls_depth=2, # depth of cross attention of CLS tokens to patch
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05
    )
    elif netname=="swin":
        from models.swin import swin_t
        net = swin_t(window_size=patch,
                    num_classes=num_classes,
                    downscaling_factors=(2,2,2,1))

    elif netname == "swinpool":
        from models.swin_pool3 import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size)
        #window_size =patch, num_classes=num_classes, downscaling_factors=(2,2,2,1))
        print(net.flops())
        

    elif netname == "swinpool-exp":
        from models.swin_pool3 import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size, depths=[2, 2, 2, 2])
        #window_size =patch, num_classes=num_classes, downscaling_factors=(2,2,2,1))
        print(net.flops())
        
    elif netname == "swinpooltpretrained22":
        from models.swin_pool3 import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size)
        #net.load_state_dict(torch.load("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth"))
        #net.eval()
        load_pretrained("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth", net)

    elif netname == "swinabl":
        from models.swin_abl import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size)
        #window_size =patch, num_classes=num_classes, downscaling_factors=(2,2,2,1))
        print(net.flops())

    elif netname == "swinabl-s":
        from models.swin_abl import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size, depths=[2, 2, 18, 2])
        #window_size =patch, num_classes=num_classes, downscaling_factors=(2,2,2,1))

    elif netname == "swinoff":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size)
        print(net.flops())

    elif netname == "swinoff-t-22pretrained":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size)
        for param in net.parameters():
            print(type(param), param.size())
        #net.load_state_dict(torch.load("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth"))
        #net.eval()
        load_pretrained("./pretrainedmodels/swin_tiny_patch4_window7_224_22k.pth", net)

    elif netname == "swinoff-s":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size, depths = [2,2,18,2])
    
    elif netname =="swinoff-b":
        from models.swin_official import SwinTransformer
        net = SwinTransformer(window_size=patch, num_classes=num_classes, img_size=size, depths = [2,2,18,2], embed_dim = 128, num_heads = (4,8,16,32))
    

    elif netname == "swinoff2":
        from models.swin_official2 import SwinTransformerV2
        net = SwinTransformerV2(window_size=patch, num_classes=num_classes, img_size=size)

    elif netname == "vit_mlp":
        from models.vit_MLP import ResMLP
        net = ResMLP(dim = int(dimhead), num_classes=num_classes, patch_size=patch, image_size=size, depth=16, mlp_dim=512, in_channels=3)

    elif netname == "poolformer":
        from models.vit_pool import PoolFormer
        net = PoolFormer(layers=[2, 2, 6, 2], embed_dims=[64, 128, 320, 512], mlp_ratios= [4, 4, 4, 4], downsamples =[True, True, True, True]);

    if cuda:
	    net = torch.nn.DataParallel(net).cuda()
    else:
        net = torch.nn.DataParallel(net)

    return net
