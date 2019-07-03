from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import numpy as np


class ImpressionFeatureCleaned(FeatureBase):

    """
    cleaned accomodation features with the formula:
    
    #times_clicked/(#times_appered + Shrink)

    item_id | properties
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'impression_features_cleaned'
        super(ImpressionFeatureCleaned, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        from extract_features.impression_features import ImpressionFeature
        o = ImpressionFeature(self.mode)
        f = o.read_feature(True)
        filterd = f[['item_id', 'properties3 Star', 'properties4 Star',
            'propertiesAccessible Hotel', 'propertiesAccessible Parking',
            'propertiesAir Conditioning', 'propertiesBathtub',
            'propertiesBeach', 'propertiesBeauty Salon',
            'propertiesBike Rental', 'propertiesBoat Rental',
            'propertiesBody Treatments', 'propertiesBusiness Centre',
            'propertiesBusiness Hotel', 'propertiesCable TV',
            'propertiesCar Park', 'propertiesCentral Heating',
            'propertiesChildcare', 'propertiesComputer with Internet',
            'propertiesConcierge', 'propertiesConference Rooms',
            'propertiesConvenience Store', 'propertiesConvention Hotel',
            'propertiesCosmetic Mirror', 'propertiesCot',
            'propertiesDeck Chairs', 'propertiesDesk',
            'propertiesDirect beach access', 'propertiesDiving',
            'propertiesElectric Kettle',
            'propertiesExpress Check-In / Check-Out',
            'propertiesFamily Friendly', 'propertiesFitness',
            'propertiesFlatscreen TV', 'propertiesFree WiFi (Combined)',
            'propertiesFree WiFi (Public Areas)',
            'propertiesFree WiFi (Rooms)', 'propertiesFrom 2 Stars',
            'propertiesFrom 3 Stars', 'propertiesFrom 4 Stars',
            'propertiesGolf Course', 'propertiesGood Rating', 'propertiesGym',
            'propertiesHairdresser', 'propertiesHairdryer',
            'propertiesHiking Trail', 'propertiesHoneymoon',
            'propertiesHorse Riding', 'propertiesHot Stone Massage',
            'propertiesHotel', 'propertiesHotel Bar',
            'propertiesIroning Board', 'propertiesJacuzzi (Hotel)',
            "propertiesKids' Club", 'propertiesLarge Groups',
            'propertiesLaundry Service', 'propertiesLift',
            'propertiesLuxury Hotel', 'propertiesMassage',
            'propertiesMinigolf', 'propertiesNightclub',
            'propertiesNon-Smoking Rooms',
            'propertiesOn-Site Boutique Shopping',
            'propertiesOpenable Windows', 'propertiesOrganised Activities',
            'propertiesPet Friendly', 'propertiesPlayground',
            'propertiesPool Table', 'propertiesPorter', 'propertiesRadio',
            'propertiesReception (24/7)', 'propertiesResort',
            'propertiesRestaurant', 'propertiesRomantic',
            'propertiesRoom Service', 'propertiesRoom Service (24/7)',
            'propertiesSafe (Hotel)', 'propertiesSafe (Rooms)',
            'propertiesSailing', 'propertiesSatellite TV',
            'propertiesSatisfactory Rating', 'propertiesSauna',
            'propertiesShower', 'propertiesSingles',
            'propertiesSitting Area (Rooms)', 'propertiesSolarium',
            'propertiesSpa (Wellness Facility)', 'propertiesSpa Hotel',
            'propertiesSteam Room', 'propertiesSun Umbrellas',
            'propertiesSurfing', 'propertiesSwimming Pool (Bar)',
            'propertiesSwimming Pool (Combined Filter)',
            'propertiesSwimming Pool (Indoor)',
            'propertiesSwimming Pool (Outdoor)', 'propertiesTable Tennis',
            'propertiesTelephone', 'propertiesTelevision',
            'propertiesTennis Court', 'propertiesTerrace (Hotel)',
            'propertiesTowels', 'propertiesVolleyball',
            'propertiesWheelchair Accessible', 'propertiesWiFi (Public Areas)',
            'propertiesWiFi (Rooms)']]
        
        justwifi = filterd.filter(regex='WiFi', axis=1)
        filterd['properties WiFi'] = np.zeros(len(filterd), dtype=np.uint8)
        filterd['properties WiFi'][justwifi.any(axis=1)] = 1
        filterd = filterd.drop(justwifi.columns, axis=1)

        justpool = filterd.filter(regex='Swimming Pool', axis=1)
        filterd['properties Swimming Pool'] = np.zeros(len(filterd), dtype=np.uint8)
        filterd['properties Swimming Pool'][justpool.any(axis=1)] = 1
        filterd = filterd.drop(justpool.columns, axis=1)

        return filterd

if __name__ == '__main__':
    from utils.menu import mode_selection, cluster_selection

    cluster = cluster_selection()
    mode = mode_selection()
    c = ImpressionFeatureCleaned(mode=mode, cluster=cluster)
    c.save_feature()
