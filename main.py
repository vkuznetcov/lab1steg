import cv2
import numpy

# GREEN-2 XOR RED-1_____BLUE_____3.12
r_coef = 77 / 256
g_coef = 150 / 256
b_coef = 29 / 256

# Task 1,2
green_bit_plane_number = 2
red_bit_plane_number = 1

green_significance = g_coef * green_bit_plane_number
red_significance = r_coef * red_bit_plane_number

# Task 3,4
sigma = 4 + 4 * (15 % 3)
noise = numpy.random.randint(0, sigma - 1, size=(512, 512)).astype(numpy.uint8)


# Getting color channel
def get_color_channel(image, channel):
    blue, green, red = cv2.split(image)
    if channel == 'blue':
        return blue
    elif channel == 'red':
        return red
    elif channel == 'green':
        return green


# Getting bit plane
def get_bit_plane(image, place_num):
    return image & (2 ** (place_num - 1))


# Image embedding method 2
def ies4_encode(original_image, watermark_image):
    blue_channel = get_color_channel(original_image, 'blue')
    green_channel = get_color_channel(original_image, 'green')
    red_channel = get_color_channel(original_image, 'red')
    blue_channel_with_watermark = ((blue_channel // (2 * sigma)) * 2 * sigma + get_color_channel(watermark_image,
                                                                                                 'blue') * sigma + noise)
    merged = cv2.merge([blue_channel_with_watermark, green_channel, red_channel])
    return blue_channel_with_watermark, merged


# Watermark decoding method 2
def ies4_decode(image_with_watermark, original_image):
    channel_with_watermark = get_color_channel(image_with_watermark, 'blue')
    original_channel = get_color_channel(original_image, 'blue')
    return channel_with_watermark - (original_channel // (2 * sigma) * 2 * sigma) - noise


# Watermark decoding method 1
def ies1_decode(encode_image, first_bit_plain, second_bit_plain
                , first_color_channel, second_color_channel):
    # second_channel = get_color_channel(baboon, second_color_channel)
    # second_bit_plain = get_bit_place(second_channel, second_bit_place_number)

    first_channel = get_color_channel(encode_image, first_color_channel)
    first_bit_place = get_bit_plane(first_channel, first_bit_plain)

    return first_bit_place * 255


# Image embedding method 1
def ies1_encode(original_image, watermark_image, first_color_channel, first_bit_plain,
                second_color_channel, second_bit_plain):
    clear_bit_place = 255 - (2 ** (first_bit_plain - 1))
    watermark_bit_plain = ((watermark_image / 255) * (2 ** (first_bit_plain - 1))).astype(numpy.uint8)
    binary_watermark = get_color_channel(watermark_bit_plain, first_color_channel)
    # if clear_bit_place == 254: 254 = 11111110, зануляем 1-ю битовую плоскость
    channel_with_empty_bit_plain = get_color_channel(original_image, first_color_channel) & clear_bit_place
    channel_with_watermark = channel_with_empty_bit_plain | binary_watermark

    second_channel = get_color_channel(original_image, second_color_channel)
    second_bit_place = get_bit_plane(second_channel, second_bit_plain)
    channel_result = second_bit_place ^ channel_with_watermark

    r = get_color_channel(original_image, 'red')
    g = get_color_channel(original_image, 'green')
    b = get_color_channel(original_image, 'blue')

    if first_color_channel == 'blue':
        return channel_result, cv2.merge([channel_result, g, r])
    if first_color_channel == 'red':
        return channel_result, cv2.merge([b, g, channel_result])
    if first_color_channel == 'green':
        return channel_result, cv2.merge([b, channel_result, r])


if __name__ == '__main__':
    baboon = cv2.imread('baboon.tif')
    watermark = cv2.imread('ornament.tif')
    result_ies1 = None
    decoded_ies1 = None
    channel_with_watermark_ies1 = None

    result_ies4 = None
    decoded_ies4 = None
    channel_with_watermark_ies4 = None

# Deciding what bit plane we are going to embed watermark for 1 method
    if red_significance > green_significance:
        channel_with_watermark_ies1, result_ies1 = ies1_encode(baboon, watermark, 'green', green_bit_plane_number,
                                                               'red',
                                                               red_bit_plane_number)
        decoded_ies1 = ies1_decode(result_ies1, green_bit_plane_number, red_bit_plane_number, 'green', 'red')
    else:
        channel_with_watermark_ies1, result_ies1 = ies1_encode(baboon, watermark, 'red', red_bit_plane_number, 'green',
                                                               green_bit_plane_number)
        decoded_ies1 = ies1_decode(result_ies1, red_bit_plane_number, green_bit_plane_number, 'red', 'green')

# Results for 2 method
    channel_with_watermark_ies4, result_ies4 = ies4_encode(baboon, watermark)
    decoded_ies4 = ies4_decode(result_ies4, baboon)

# Images showing
    cv2.imshow('encoded ies1', result_ies1)
    cv2.imshow('channel ies1', channel_with_watermark_ies1)
    cv2.imshow('watermark ies1', decoded_ies1)
    cv2.waitKey(0)


    cv2.imshow('encoded ies4', result_ies4)
    cv2.imshow('channel ies4', channel_with_watermark_ies4)
    cv2.imshow('watermark ies4', decoded_ies4)
    cv2.waitKey(0)
