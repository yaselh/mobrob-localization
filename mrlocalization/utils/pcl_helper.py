import struct

def rgb_to_float(r, g, b):
     rgb = r << 16 | g << 8 | b
     # p.rgb = *reinterpret_cast<float*>(&rgb);
     rgb_bytes = struct.pack('I', rgb)
     p_rgb = struct.unpack('f', rgb_bytes)[0]
     return p_rgb


def float_to_rgb(p_rgb):
     # rgb = *reinterpret_cast<int*>(&p.rgb)
     rgb_bytes = struct.pack('f', p_rgb)
     rgb = struct.unpack('I', rgb_bytes)[0]

     r = (rgb >> 16) & 0x0000ff
     g = (rgb >> 8)  & 0x0000ff
     b = (rgb)       & 0x0000ff
     return r,g,b

def filter_far_points(pointcloud):
    # Filter all points that are more than x meters away
    fil = pointcloud.make_passthrough_filter()
    fil.set_filter_field_name("z")
    fil.set_filter_limits(0, 2.0)
    filtered_pointcloud = fil.filter()
    return filtered_pointcloud

if __name__ == '__main__':
    print(rgb_to_float(255, 0, 0))
