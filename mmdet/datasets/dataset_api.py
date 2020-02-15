from lvis import LVIS
from pycocotools.coco import COCO


class CocoAPI:
    """
    General API for COCO json style annotation
    """

    def __init__(self, anno_file, api_type):
        assert api_type in ['COCO', 'LVIS']
        if api_type == 'COCO':
            api = COCO(anno_file)
        elif api_type == 'LVIS':
            api = LVIS(anno_file)
        else:
            raise NotImplementedError

        self.api_type = api_type
        self.api = api

    @property
    def dataset(self):
        return self.api.dataset

    @property
    def anns(self):
        return self.api.anns

    @property
    def cats(self):
        return self.api.cats

    @property
    def imgs(self):
        return self.api.imgs

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None):
        if self.api_type == 'COCO':
            img_ids = [] if img_ids is None else img_ids
            cat_ids = [] if cat_ids is None else cat_ids
            area_rng = [] if area_rng is None else area_rng
            ann_ids = self.api.getAnnIds(
                imgIds=img_ids, catIds=cat_ids, areaRng=area_rng)
        elif self.api_type == 'LVIS':
            ann_ids = self.api.get_ann_ids(
                img_ids=img_ids, cat_ids=cat_ids, area_rng=area_rng)
        else:
            raise NotImplementedError
        return ann_ids

    def get_cat_ids(self):
        if self.api_type == 'COCO':
            cat_ids = self.api.getCatIds()
        elif self.api_type == 'LVIS':
            cat_ids = self.api.get_cat_ids()
        else:
            raise NotImplementedError
        return cat_ids

    def get_img_ids(self):
        if self.api_type == 'COCO':
            img_ids = self.api.getImgIds()
        elif self.api_type == 'LVIS':
            img_ids = self.api.get_img_ids()
        else:
            raise NotImplementedError
        return img_ids

    def load_anns(self, ids=None):
        if self.api_type == 'COCO':
            ids = [] if ids is None else ids
            anns = self.api.loadAnns(ids=ids)
        elif self.api_type == 'LVIS':
            anns = self.api.load_anns(ids=ids)
        else:
            raise NotImplementedError
        return anns

    def load_cats(self, ids):
        if self.api_type == 'COCO':
            ids = [] if ids is None else ids
            cats = self.api.loadCats(ids=ids)
        elif self.api_type == 'LVIS':
            cats = self.api.load_cats(ids=ids)
        else:
            raise NotImplementedError
        return cats

    def load_imgs(self, ids):
        if self.api_type == 'COCO':
            ids = [] if ids is None else ids
            imgs = self.api.loadImgs(ids=ids)
        elif self.api_type == 'LVIS':
            imgs = self.api.load_imgs(ids=ids)
        else:
            raise NotImplementedError
        return imgs

    def download(self, *args, **kwargs):
        self.api.download(*args, **kwargs)
