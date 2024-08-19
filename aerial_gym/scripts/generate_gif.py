import imageio

file_path = '/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/camera_output/frames/'

with imageio.get_writer(uri='/home/cgv841/wzm/FYP/AGAPG/aerial_gym/scripts/camera_output/test.gif', mode='I', fps=10) as writer:
    for i in range(99):
        if i:
            writer.append_data(imageio.imread(file_path + f'tmp{i+1}0.png'))
        else:
            writer.append_data(imageio.imread(file_path + 'tmp0.png'))

print("GIF Generate Complete!")