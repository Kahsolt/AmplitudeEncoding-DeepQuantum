![amp_enc_vqc.png](./amp_enc_vqc.png)

```
[QAM256] 大端序
11000001 10000001 10010001 11010001 11010101 10010101 10000101 11000101 | 11000100 10000100 10010100 11010100 11010000 10010000 10000000 11000000
01000001 00000001 00010001 01010001 01010101 00010101 00000101 01000101 | 01000100 00000100 00010100 01010100 01010000 00010000 00000000 01000000
01100001 00100001 00110001 01110001 01110101 00110101 00100101 01100101 | 01100100 00100100 00110100 01110100 01110000 00110000 00100000 01100000
11100001 10100001 10110001 11110001 11110101 10110101 10100101 11100101 | 11100100 10100100 10110100 11110100 11110000 10110000 10100000 11100000
11101001 10101001 10111001 11111001 11111101 10111101 10101101 11101101 | 11101100 10101100 10111100 11111100 11111000 10111000 10101000 11101000
01101001 00101001 00111001 01111001 01111101 00111101 00101101 01101101 | 01101100 00101100 00111100 01111100 01111000 00111000 00101000 01101000
01001001 00001001 00011001 01011001 01011101 00011101 00001101 01001101 | 01001100 00001100 00011100 01011100 01011000 00011000 00001000 01001000
11001001 10001001 10011001 11011001 11011101 10011101 10001101 11001101 | 11001100 10001100 10011100 11011100 11011000 10011000 10001000 11001000
------------------------------------------------------------------------+------------------------------------------------------------------------
11001011 10001011 10011011 11011011 11011111 10011111 10001111 11001111 | 11001110 10001110 10011110 11011110 11011010 10011010 10001010 11001010
01001011 00001011 00011011 01011011 01011111 00011111 00001111 01001111 | 01001110 00001110 00011110 01011110 01011010 00011010 00001010 01001010
01101011 00101011 00111011 01111011 01111111 00111111 00101111 01101111 | 01101110 00101110 00111110 01111110 01111010 00111010 00101010 01101010
11101011 10101011 10111011 11111011 11111111 10111111 10101111 11101111 | 11101110 10101110 10111110 11111110 11111010 10111010 10101010 11101010
11100011 10100011 10110011 11110011 11110111 10110111 10100111 11100111 | 11100110 10100110 10110110 11110110 11110010 10110010 10100010 11100010
01100011 00100011 00110011 01110011 01110111 00110111 00100111 01100111 | 01100110 00100110 00110110 01110110 01110010 00110010 00100010 01100010
01000011 00000011 00010011 01010011 01010111 00010111 00000111 01000111 | 01000110 00000110 00010110 01010110 01010010 00010010 00000010 01000010
11000011 10000011 10010011 11010011 11010111 10010111 10000111 11000111 | 11000110 10000110 10010110 11010110 11010010 10010010 10000010 11000010
```

Assume an 8x8 RGB image is embeded in a quantunm circuit with AmpEnc in layout:
```
| B | R |
| 0 | G |
```

To simulate a classical CNN, we'd firstly find a way to convolve over all `2x2` blocks (aggregating from locality!). There are $3*8*8/(2*2) = 48$ valid sub-blocks, hence in analogue we'll need $48$ `mult-controlled U3` gates to extract the features. Note that in classical CNNs, the filter kernels (ie. learnable parameters) are shared across all sub-blocks. But in NISQ, this is kinda difficult to mimic... OK, we can also just try making them different. 

Now the second question, where to put and how to arrange the feature map? The classical manner is to allocate a new memory block and keep it shaped as a 3d-array. Again, NISQ simulator cannot mimic this due to complexity... Experimentally we use $4$ ancilla qubits to store the feature values, lest polluting the original information holding in `|enc>` :) The ancilla-qubit use order follows the image spatial structure: left-upper blocks uses teh 1st ancilla qubit, right-upper the 2nd, etc. So that the i-th aniclla qubit collects all data from the same region within the image. 

To conclude so far, we utilize this ansatz template for **psuedo-convolutional feature extraction**: 

```
 |enc>            B-ch              R-ch       G-ch
Channel |x>--●-●-●-●-●-●-●-●-------○-○-○-○--
Channel |x>--○-○-○-○-○-○-○-○-------○-○-○-○--
Spatial |x>--○-○-●-●-○-○-●-●-------●-●-○-○--
Spatial |x>--○-○-○-○-○-○-○-○--...--○-○-○-○--    ...
Spatial |x>--○-●-●-○-○-●-●-○-------○-●-●-○--
Spatial |x>--○-○-○-○-●-○-●-●-------○-○-○-○--
Spatial |x>--|-|-|-|-|-|-|-|-------|-|-|-|--
Spatial |x>--|-|-|-|-|-|-|-|-------|-|-|-|--
 |anc>       | | | | | | | |  ...  | | | |
Ancilla |0>--U-U-|-|-U-U-|-|-------U-U-|-|--  // ↖：collects info from left  upper  region of the image 
Ancilla |0>------U-U-----U-U-----------U-U--  // ↗：collects info from right upper  region of the image 
Ancilla |0>---------------------------------  // ↙：collects info from left  bottom region of the image 
Ancilla |0>---------------------------------  // ↘：collects info from right bottom region of the image 
```

Actually, it's just a $PREP-SEL$ structure that reads from an `AmpEnc` register, then writes to an `AngleEnc` register, ie. the `AmpEnc->AngleEnc` converter. 😮 

Then we need a **learnable feature transformation**, just wrap a Hardware Efficient Ansatz around `|anc>` should look fine. 

```
                | | | |       (pairwise)
Ancilla |0>-U3--U-|-|-|-------U3-CU3--
Ancilla |0>-U3----U-|-|--...--U3-CU3--
Ancilla |0>-U3------U-|--...--U3-CU3--
Ancilla |0>-U3--------U-------U3-CU3--
```

Another key concept in classical CNN is **pooling**, a hardcoded non-learnable function force reducing the feature map size. That suggests us to do the same thing in `|anc>` register... uhm... but if so, we'll need anthor `|anc2>` to store the new results, wtf??! 

No! You'd no need pooling layers!! 
Since that "the i-th ancilla qubit collects all data from the same region grid within the image", this is by nature a "sum" pooler!  
We just need the rot-param divided by `n_feat_maps` to turn it into an "avg" pooler, keeping the power equivalent! 🎉  

Now it's time to chain up several convolutional layers to form a feedfoward network. This word "feedfoward" means that output feature maps from the 1st layer is fed into the input of the 2nd layer.  This is also impossible in our case -- neither `|anc>` or `|enc>` is proper for this operation. ┑(￣Д ￣)┍  

Have a try of this! Like above, each `mctr-U3` group extract features from a different resolution stage (from fine to coarse):

```
                B/R/G-ch
|enc>              x4                  x2                  x1
  Ch  |x>---------[○/●]---------------[○/●]---------------[○/●]---
  Ch  |x>---------[○/●]---------------[○/●]---------------[○/●]---
  Sp  |x>---------[○/●]---------------[○/●]-----------------|-----
  Sp  |x>---------[○/●]---------------[○/●]-----------------|-----
  Sp  |x>---------[○/●]-----------------|-------------------|-----
  Sp  |x>---------[○/●]-----------------|-------------------|-----
  Sp  |x>-----------|-------------------|-------------------|-----
  Sp  |x>-----------|-------------------|-------------------|-----
                    |                   |                   |
|anc> |0>-[U3]--[mctr-U3]-[U3-CU3]--[mctr-U3]-[U3-CU3]--[mctr-U3]---[CU3-U3]-->[Measure(Z)]
      |0>----------------------------------------------------[U3]/    // add a new wire to meet num-class
Notes:          U3(θ_i/48)          U3(θ_i/12)          U3(θ_i/3)
               (48 mctrls)         (12 mctrls)         (3 mctrls)
                2x2 block           4x4 block           8x8 block
```

This total gate_cnt  is $(48+12+3)   + 4*8 = 95$ for `n_layer=1` of this ansatz.
This total param_cnt is $(48+12+3)/3 + 4*8 = 53$ for `n_layer=1` of this ansatz.

**TODO**: Can we theoretically proove if this is a kind of ResNet, Feature-Pyramid or just Data-ReUpload in math? 🤔

Finall for the measure part, you just need enough tuning and experiments...


<div STYLE="page-break-after: always;"></div>


⚪ Extending to the real CIFAR10 problem

We have a $12$ qubits `|enc>` storing in 4096-QAM. The sub-block counts:

- `2x2`: 768
- `4x4`: 192
- `8x8`: 48
- `16x16`: 12
- `32x32`: 3

And still allocate only $4$ qubits as `|anc>`, to satisfy the implicit "avg" pooler.
