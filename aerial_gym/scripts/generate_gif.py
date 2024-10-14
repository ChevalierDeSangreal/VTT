import imageio

file_path = '/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/frames/'

with imageio.get_writer(uri='/home/wangzimo/VTT/VTT/aerial_gym/scripts/camera_output/test.gif', mode='I', fps=50) as writer:
    for i in range(529):
        if i:
            writer.append_data(imageio.imread(file_path + f'{i}.png'))

print("GIF Generate Complete!")