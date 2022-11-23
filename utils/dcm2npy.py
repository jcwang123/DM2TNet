import glob
import os
import time
import skimage.draw
import nibabel as nib

import dicom as dicom

# import pydicom as dicom
import numpy as np
import SimpleITK as sitk
from shapely.geometry.polygon import Polygon

def get_current_dir():
    return os.path.dirname(__file__)

def get_roi_mapping(roi_names_path):
    roi_to_index = {}
    name = None
    index = -1
###########################
    for line in open(roi_names_path).readlines():
        line = line.rstrip('\n')
        if not line.startswith('\t') and not line.startswith(' '):
            name = line
            index += 1
        else:
            start = line.find('\'')
            end = line.rfind('\'')
            line = line[start+1:end]
            roi_to_index[line] = {'name': name, 'index': index}
    return roi_to_index


class Series(object):
    def __init__(self, dir, roi_file_path=None):
        self.series_dir = dir
        self.roi_file_path = roi_file_path
        print('Process series {}'.format(self.series_dir))
        self.init_rt()
        self.init_data()
        self.init_label()

    def init_rt(self):
        dcm_files = []
        rt_files = []
        
        for fn in os.listdir(self.series_dir):
            file_path = os.path.join(self.series_dir, fn)
            if not file_path.lower().endswith('dcm'):
                print('\t\tWarn, {} is not a dicom file'.format(file_path))
                continue
            dcm = dicom.read_file(file_path, force=True)
            try:
                if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':
                    rt_files.append(file_path)
                elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.2':
                    dcm_files.append(file_path)
                else:
                    print('\t\tWarn, {} is not rtss or ct file'.format(file_path))
            except:
                rt_files.append(file_path)
                continue
        print('\tFind {} ct files, {} rtss files'.format(len(dcm_files), len(rt_files)))
        self.dcm_paths = dcm_files
        if len(rt_files) > 1:
            print('\t\tMore than one rtss files found')
            self.rt_path = -1
            return
        self.rt_path = rt_files[0]

    def init_data(self):
        reader = sitk.ImageSeriesReader()
        dcm_files = reader.GetGDCMSeriesFileNames(self.series_dir)
        reader.SetFileNames(dcm_files)
        reader.SetLoadPrivateTags(True)
        slices = reader.Execute()

        print('\tSlices Spacing', slices.GetSpacing())
        print('\tSlices Direction', slices.GetDirection())
        if slices.GetDirection()[0] == 1:
            assert slices.GetDirection()[0] == 1 and slices.GetDirection()[4] == 1 and slices.GetDirection()[8] == 1
            self.direction_up = True
        else:
            assert slices.GetDirection()[0] == -1 and slices.GetDirection()[4] == -1 and slices.GetDirection()[8] == 1
            self.direction_up = False
        print('\tSlices Origin', slices.GetOrigin())
        self.slices = slices
        self.data = sitk.GetArrayFromImage(self.slices)

    @property
    def spacing(self):
        return self.slices.spacing

    @property
    def direction(self):
        return self.slices.direction

    @property
    def origin(self):
        return self.slices.origin

    def init_label(self):
        if self.rt_path == -1:
            print('no label')
            return
        # check rtss and dicom data is of the same serie
        ct_dcm = dicom.read_file(self.dcm_paths[0], force=True)
        rt_dcm = dicom.read_file(self.rt_path, force=True)
        if hasattr(rt_dcm, 'SpecificCharacterSet') and rt_dcm.SpecificCharacterSet == 'UNAVAILABLE':
            rt_dcm.SpecificCharacterSet = 'ISO_IR 100'
            rt_dcm.save_as(self.rt_path)
            rt_dcm = dicom.read_file(self.rt_path, force=True)
        elif not hasattr(rt_dcm, 'SpecificCharacterSet'):
            rt_dcm.SpecificCharacterSet = 'ISO_IR 100'
            rt_dcm.save_as(self.rt_path)
            rt_dcm = dicom.read_file(self.rt_path, force=True)
#         ttt = 0
#         for i in range(len(rt_dcm.ReferencedFrameOfReferenceSequence)):
#             if rt_dcm.ReferencedFrameOfReferenceSequence[i].FrameOfReferenceUID == ct_dcm.FrameOfReferenceUID:
#                 ttt = 1
              
#         assert ttt==1, 'rtss have different frameOfReferenceUID with ct files'

        roi_mapping = get_roi_mapping(self.roi_file_path)
        total_rois = len(set([t['index'] for t in roi_mapping.values()]))

        index_to_map_index = {}
        for ss_roi in rt_dcm.StructureSetROISequence:
            roi_index = int(ss_roi.ROINumber)
            roi_name = str(ss_roi.ROIName).lower()
            if roi_name not in roi_mapping:
                print('\tSkip roi [{}]'.format(roi_name))
                continue
            roi_map_index = roi_mapping[roi_name]['index']
            index_to_map_index[roi_index] = roi_map_index

        # roi_contours = {}
        H, W, Z = self.slices.GetSize()
        # print(self.slices.GetSize())
        label = np.zeros((Z, H, W, total_rois), dtype=np.float32)

        for roi_contour in rt_dcm.ROIContourSequence:
            ref_roi_number = int(roi_contour.ReferencedROINumber)
            if ref_roi_number not in index_to_map_index:
                continue

            contours_each_slice = [(i, []) for i in range(Z)]

            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    contour_data = contour.ContourData
                    assert len(contour_data) % 3 == 0
                    contour_data = np.array(contour_data).reshape((-1, 3))
                    contour_data = [self.slices.TransformPhysicalPointToContinuousIndex(t) for t in contour_data]
                    contour_data = np.array(contour_data)
                    if contour_data.shape[0] < 3:
                        print('\t\tless than 3 points in contour', contour_data.shape)
                        continue
                    slice_index = round(contour_data[0, 2], 0)
                    # print(slice_index, round(slice_index, 0))
                    assert abs(slice_index - int(slice_index)) < 1e-5
                    contours_each_slice[int(slice_index)][1].append(contour_data)

                for slice_index, contours in contours_each_slice:
                    polys = [(cont_index, Polygon(cont)) for cont_index, cont in enumerate(contours)]
                    inner_polys = []
                    for poly_index, poly in polys:
                        for poly_index_j, poly_j in polys:
                            if poly_index_j != poly_index and poly_j.contains(poly):
                                inner_polys.append(poly_index)
                                break

                    for contour in contours:
                        rr, cc = skimage.draw.polygon(contour[:, 1], contour[:, 0])
                        label[slice_index, rr, cc, index_to_map_index[ref_roi_number]] = 1
                    for cont_index in inner_polys:
                        rr, cc = skimage.draw.polygon(contours[cont_index][:, 1], contours[cont_index][:, 0])
                        label[slice_index, rr, cc, index_to_map_index[ref_roi_number]] = 0
        
        #self.data = sitk.GetArrayFromImage(self.slices)
        self.label = label