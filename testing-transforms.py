from  nipype.interfaces.ants import ApplyTransforms

baseDir = '/home/jms565/Research/CHP-PIRC/markov-movement-correction/0003_MR1_18991230_000000EP2DBOLDLINCONNECTIVITYs004a001/'
template = baseDir + 'testing/000.nii.gz'
moving = baseDir + 'timepoints/024.nii.gz'
outputImg = baseDir + 'testing/024_applied_mat.nii.gz'
affineTransform = baseDir + 'testing/testing_transform_0GenericAffine.mat'
warpedTransform = baseDir + 'testing/testing_transform_1InverseWarp.nii.gz'

at = ApplyTransforms()
at.inputs.input_image = moving
at.inputs.reference_image = moving
at.inputs.output_image = outputImg
at.inputs.transforms = affineTransform
at.inputs.interpolation = 'NearestNeighbor'
at.inputs.invert_transform_flags = [False]
at.run()

# at = ApplyTransforms()
# at.inputs.input_image = outputImg
# at.inputs.reference_image = outputImg
# at.inputs.output_image = outputImg
# at.inputs.transforms = affineTransform
# at.inputs.interpolation = 'NearestNeighbor'
# at.inputs.invert_transform_flags = [False]
# at.run()
