# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.udaneck_dacs import UDANeck_DACS
from mmseg.models.uda.partially_frooze_ema_dacs import PartiallyFroozeEmaDacs
__all__ = ['DACS', 'UDANeck_DACS', 'PartiallyFroozeEmaDacs']
