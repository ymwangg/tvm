from collections import defaultdict
import re
import math

class CudaCode:
    def __init__(self):
        self.for_variant = 0

    def print(self):
        print("for_variant is: ", self.for_variant)


class BinaryCode:
    def __init__(self, input_name):
        self.name = input_name
        self.add = 0
        self.mov = 0
        self.st_local = 0
        self.st_shared = 0
        self.st_global = 0
        self.ld_local = 0
        self.ld_global = 0
        self.ld_shared = 0
        self.mul = 0
        self.mad = 0
        self.fma = 0
        self.shl = 0
        self.shr = 0
        self._or = 0
        self.sub = 0
        self.ld_param = 0
        self.cvta = 0
        self.setp = 0
        self._and = 0
        self._not = 0
    
    def __lt__(self, b):
        return self.name < b.name

    def print(self):
        temp_map = {'add': self.add,
                    'mov': self.mov,
                    'mul': self.mul,
                    'mad': self.mad,
                    'st_local': self.st_local,
                    'st_shared': self.st_shared,
                    'st_global': self.st_global,
                    'ld_local': self.ld_local,
                    'ld_global': self.ld_global,
                    'ld_shared': self.ld_shared,
                    'fma': self.fma,
                    'shl': self.shl,
                    'shr': self.shr,
                    '_or': self._or,
                    'sub': self.sub,
                    'ld_param': self.ld_param,
                    'cvta': self.cvta,
                    'setp': self.setp,
                    '_and': self._and,
                    '_not': self._not,
                }
        print('code: ', self.name) 
        for operator_name in temp_map:
            if temp_map[operator_name] is not 0:
                print("{} : {}".format(operator_name, temp_map[operator_name]))

    def __add__(self, b):
        self.add += b.add
        self.mov += b.mov
        self.st_local += b.st_local
        self.st_shared += b.st_shared
        self.st_global += b.st_global
        self.ld_local += b.ld_local
        self.ld_global += b.ld_global
        self.ld_shared += b.ld_shared
        self.mul += b.mul
        self.mad += b.mad
        self.fma += b.fma
        self.shl += b.shl
        self.shr += b.shr
        self._or += b._or
        self.sub += b.sub
        self.ld_param += b.ld_param
        self.cvta += b.cvta
        self.setp += b.setp
        self._and += b._and
        self._not += b._not

        return self

    def __mul__(self, b):
        self.add *= b
        self.mov *= b
        self.st_local *= b
        self.st_shared *= b
        self.st_global *= b
        self.ld_local *= b
        self.ld_global *= b
        self.ld_shared *= b
        self.mul *= b
        self.mad *= b
        self.fma *= b
        self.shl *= b
        self.shr *= b
        self._or *= b
        self.sub *= b
        self.ld_param *= b
        self.cvta *= b
        self.setp *= b
        self._and *= b
        self._not *= b

        return self

    def clear(self):
        self.add = 0
        self.mov = 0
        self.st_local = 0
        self.st_shared = 0
        self.st_global = 0
        self.ld_local = 0
        self.ld_global = 0
        self.ld_shared = 0
        self.mul = 0
        self.mad = 0
        self.fma = 0
        self.shl = 0
        self.shr = 0
        self._or = 0
        self.sub = 0
        self.ld_param = 0
        self.cvta = 0
        self.setp = 0
        self._and = 0
        self._not = 0


class BasicBlock:
    def __init__(self, input_name):
        self.name = input_name
        self.codelist = list()
        self.forloop = False
        self.end_for = False

    def print_block(self):
        print('BLOCK: ', self.name)
        print("self.forloop: ", self.forloop)
        print("self.end_for: ", self.end_for)
        for code in self.codelist:
            code.print()

def extract_feature(line, code):
    if '.s64' in line or '.f64' in line:
        #print(line)
        return
    if 'add.' in line:
        code.add += 1
    if 'mov' in line:
        code.mov += 1
    if 'st.local' in line:
        code.st_local += 1
    if 'st.shared' in line:
        code.st_shared += 1
    if 'st.global' in line:
        code.st_global += 1
    if 'ld.local' in line:
        code.ld_local += 1
    if 'ld.global' in line:
        code.ld_global += 1
    if 'ld.shared' in line:
        code.ld_shared += 1
    if 'mul' in line:
        code.mul += 1
    if 'mad' in line:
        code.mad += 1
    if 'fma' in line:
        code.fma += 1
    if 'shl' in line:
        code.shl += 1
    if 'shr' in line:
        code.shr += 1
    if 'or' in line:
        code._or += 1
    if 'sub' in line:
        code.sub += 1
    if 'ld.param' in line:
        code.ld_param += 1
    if 'setp' in line:
        code.setp += 1
    if 'cvta' in line:
        code.cvta += 1
    if 'and' in line:
        code._and += 1
    if 'not' in line:
        code._not += 1

def extract_register(line, register_map):
    if 'mov' in line:
        m = re.search('%r\d+, .+;', line)
        if m:
            register = m.group(0).split(',')[0]
            rightside = m.group(0).split(', ')[1].split(';')[0]
            if rightside in register_map:
                rightside = register_map[rightside]
            register_map[register] = rightside

def merge(code, code2, loop_variant):
    code = (code + code2) * loop_variant
    return code

def analysis(lines, cuda_list, bb_idx, register_map):
    # entry is a root for code block, we can dod some analysis for 'for' and 'if'
    # print("--------")
    # print(lines)
    # print("--------")
    seen_bb_list = []
    for_bb_list = []
    edgecase_list = []
    if_bb_list = []
    p_map = defaultdict(list)
    bb_p_map = defaultdict()
    for line in lines:
        # parse %p in line
        if '%p' in line:
            temp = line.split()
            if '%p' in temp[1]:
                p_map[temp[1].split(',')[0]] = [temp[2].split(',')[0], temp[3].split(';')[0]]
        extract_register(line, register_map)
        if line.startswith("BB"+str(bb_idx)):
            m = re.search('BB'+str(bb_idx)+'_\d+', line)
            seen_bb_list.append(m.group(0))
        elif "BB"+str(bb_idx) in line:
            m = re.search('BB'+str(bb_idx)+'_\d+', line)
            if m.group(0) not in seen_bb_list:
                if_bb_list.append(m.group(0))
            elif m.group(0) not in if_bb_list:
                for_bb_list.append(m.group(0))
                m2 = re.search('@%p\d+', line)
                if m2:
                    bb_p_map[m.group(0)] = m2.group(0)[1:]
            if m.group(0) in seen_bb_list and m.group(0) not in for_bb_list:
                for_bb_list.append(m.group(0))
                edgecase_list.append(m.group(0))
                m2 = re.search('%p\d+', line)
                if m2:
                    bb_p_map[m.group(0)] = m2.group(0)[1:]
            
    # print("for_bb_list: ", for_bb_list)
    # print("if_bb_list: ", if_bb_list)
    # print("--------")

    for_order_map = {x:0 for x in seen_bb_list if x in for_bb_list}

    if len(for_order_map) > len(cuda_list):
        # strange case, assume internal for loop was splitted into several bb blocks
        # TODO need to identify BBs coresponding to internal loop, workaround solution for now
        # print("DANGER!!edge case: internal loop was splitted!! ")
        # print("edgecase_list: ", edgecase_list)
        if len(edgecase_list) % 2 == 1:
            # print("Failed to capture all edge cases!")
            for j in range(len(edgecase_list)):
                # TEMP logic
                for i in range(len(for_bb_list)):
                    if for_bb_list[i] == edgecase_list[j] and for_bb_list[i-1] not in edgecase_list:
                        edgecase_list.append(for_bb_list[i-1])
                        break
        # print("edgecase_list: ", edgecase_list)

    block_list = []
    # parse all for blocks
    block_idx = -1
    for line in lines:
        if line.startswith("BB"+str(bb_idx)):
            cur_block_name = re.search('BB'+str(bb_idx)+'_\d+', line).group(0)
            if cur_block_name in for_bb_list:
                # create a new block
                block = BasicBlock(cur_block_name)
                block.forloop = True
                code = BinaryCode(cur_block_name)
                block.codelist.append(code)
                block_list.append(block)
        elif 'BB'+str(bb_idx) in line:
            # print("line: ", line)
            cur_block_name = re.search('BB'+str(bb_idx)+'_\d+', line).group(0)
            if cur_block_name in for_bb_list:
                # create a new code
                code = BinaryCode(cur_block_name)
                block_list[-1].codelist.append(code)
                if cur_block_name not in edgecase_list or block_list[-1].name == cur_block_name:
                    block_list[-1].end_for = True
        else:
            extract_feature(line, block_list[-1].codelist[-1])

    # Instead of parsing loop number from cuda code, we can extract this from PTX code
    for bb_name in for_order_map:
        for_order_map[bb_name] = abs(int(register_map[p_map[bb_p_map[bb_name]][0]]) - int(p_map[bb_p_map[bb_name]][1]))

    '''
    flag = False
    prev_num = 0
    for x in for_order_map:
        if x in edgecase_list:
            if not flag:
                for_order_map[x] = cuda_list[0].for_variant
                prev_num = cuda_list[0].for_variant
                flag = True
                cuda_list.pop(0)
            else:
                for_order_map[x] = prev_num
        else:
            flag = False
            for_order_map[x] = cuda_list[0].for_variant
            cuda_list.pop(0)
    '''
    # print("for_order_map: ", for_order_map)

    # print("***************")
    # for block in block_list:
    #     block.print_block()
    # print("***************")
    
    for_idx = 0
    # merge block
    # 
    #for block in block_list:
    block_stack = []
    for block in block_list:
        if block.end_for:
            # found the close of a for loop
            num_iter = len(block.codelist) - 1
            # Merge it with itself for variant
            loop_variant = for_order_map[block.name]
            block.codelist[0] = merge(block.codelist[0], BinaryCode('tmp'), loop_variant)
            if len(block.codelist) > 0:
                block.codelist[1] = merge(block.codelist[1], block.codelist[0], 1)
                block.codelist[0].clear()
            # print('num_iter: ', num_iter)
            for i in range(1, num_iter):
                if (block_stack):
                    # print("length of block_stack: ", len(block_stack))
                    # get one from stack and merge them, assume the one we get from stack always 
                    # have only one code
                    prev_block = block_stack[-1]
                    loop_variant = for_order_map[prev_block.name]
                    block.codelist[i] = merge(block.codelist[i], prev_block.codelist[0], loop_variant)
                    prev_block.codelist[0].clear()
                    # in case left from prev blok
                    for j in range(len(prev_block.codelist) - 1):
                        block.codelist[i] = merge(block.codelist[i], prev_block.codelist[j+1], 1)
                        prev_block.codelist[j+1].clear()
                    # merge with next code
                    block.codelist[i+1] = merge(block.codelist[i+1], block.codelist[i], 1)
                    block.codelist[i].clear()

                    block_stack.pop() 

            # merge back to prev stack
            if block_stack:
                tmp_code = BinaryCode('tmp')
                for code in block.codelist:
                    tmp_code = tmp_code + code
                prev_block = block_stack[-1]
                prev_block.codelist[-1] = prev_block.codelist[-1] + tmp_code
            else:
                block_stack.append(block)
        else:
            block_stack.append(block)

    return_code = BinaryCode("ans")
    for block in block_stack:
        for code in block.codelist:
            return_code = return_code + code

    # print("--------------------")
    # return_code.print()
    # print("--------------------")

    return return_code
    
    # return only one block or all for blocks?

def analysis_global(lines, single_code, bb_idx, cuda_list):
    start_count = False
    found_end = True
    start = 0
    bb_list = []
    register_map = {}
    for i, line in enumerate(lines):
        if line.startswith("BB"+str(bb_idx)) and found_end is True:
            start_count = True
            found_end = False
            m = re.search("BB"+str(bb_idx)+'_\d+', line)
            bb_list.append(m.group(0))
            start = i
            # print(m.group(0))
        elif len((bb_list)) > 0 and bb_list[-1]+';' in line:
            found_end = True
            start_count = False
            single_code = single_code + analysis(lines[start:i+1], cuda_list, bb_idx, register_map)
            # do analysis for each root for block
        elif found_end is True and start_count is False:
            # lines within blocks or in the beginning
            # parse this
            extract_feature(line, single_code)
            extract_register(line, register_map)
    # handle edge case, where BBi_x block exist but no jump found
    if found_end == False:
        # print("edge case here BBi_x block exist but no jump found")
        for line in lines[start:]:
            extract_feature(line, single_code)

def compare_thread_map(m1, m2):
    for k in m1:
        if k in m2 and m2[k] == m1[k]:
            continue
        else:
            return False
    return True

def run(cuda_code, ptx_code, stmt):
    # seperate score/instructions per global function
    # identify how many blocks and threads
    multiplier_list = []
    thread_map = {}
    multiplier = 1
    for_count = 0
    for line in str(stmt['main'].body).split('\n'):
        if 'for (' in line or 'if (' in line or 'unrolled (' in line:
            for_count += 1
        if '}' in line:
            for_count -= 1
        if 'thread_extent = ' in line and "attr" in line and for_count == 0:
            m = re.search('blockIdx.\S', line)
            if not m:
                m = re.search('threadIdx.\S', line)
            if m.group(0) not in thread_map:
                thread_map[m.group(0)] = int(line.split('= ')[1])
        elif 'allocate' in line or 'attr' in line:
            pass
        else:
            if len(thread_map) > 0:
                if multiplier_list and not compare_thread_map(thread_map, multiplier_list[-1]):
                    # print(thread_map)
                    multiplier_list.append(thread_map)
                if not multiplier_list:
                    # print(thread_map)
                    multiplier_list.append(thread_map)
                thread_map = {}
    # print(multiplier_list)
    # helper: we can tell how many root for are
    cuda_list = []
    # only find for loop and loop variant
    # TODO move this logic into anlysis and found corresponding for during analysis
    for line in cuda_code.split("\n"):
        if 'for' in line:
            cur_for = CudaCode()
            cur_for.for_variant = int(line.split(';')[1].split('< ')[1])
            cuda_list.append(cur_for)
    # for cur_for in cuda_list:
    #     cur_for.print()

    global_start = False
    start_line = 0
    bb_idx = 0
    lines = ptx_code.split("\n")
    global_name = ""
    single_code_list = []
    for i, line in enumerate(lines):
        if '// .globl' in line:
            m = re.search('default_function_kernel\d', line)
            if m:
                global_name = m.group(0)
        # identify global functions
        if "{" in line and global_start == False:
            global_start = True
            start_line = i
        if global_start and "}" in line and "{" not in line:
            # print(global_name)
            single_code = BinaryCode(global_name)
            analysis_global(lines[start_line:i], single_code, bb_idx, cuda_list)
            global_start = False
            bb_idx = bb_idx + 1
            single_code_list.append(single_code)

    single_code_list.sort()
    # print(len(single_code_list))
    return single_code_list, multiplier_list

def cost_model(total_cmd_list, multiplier_list, resource_utilization):
    import re
    m = re.search(" \d+ registers",resource_utilization)
    registers = int(m.group(0).split()[0])
    m = re.search(" \d+ bytes smem",resource_utilization)
    smem = int(m.group(0).split()[0])
    m = re.search(" \d+ bytes cmem",resource_utilization)
    cmem = int(m.group(0).split()[0])
    # print("registers: ", registers)
    # print("smem: ", smem)
    # print("cmem: ", cmem)
    total_score = 0
    for total_cmd, multiplier_map in zip(total_cmd_list, multiplier_list):
        multiplier = 1
        thread_per_block = 1
        block_number = 1
        for x in multiplier_map:
            multiplier *= multiplier_map[x]
            if 'thread' in x:
                thread_per_block *= multiplier_map[x]
            else:
                block_number *= multiplier_map[x]
        # total_cmd.print()
        fma = total_cmd.fma
        st_local = total_cmd.st_local
        st_shared = total_cmd.st_shared
        st_global = total_cmd.st_global
        ld_local = total_cmd.ld_local
        ld_shared = total_cmd.ld_shared
        ld_global = total_cmd.ld_global
        add = total_cmd.add
        sub = total_cmd.sub
        mul = total_cmd.mul
        mad = total_cmd.mad
        _or = total_cmd._or
        _and = total_cmd._and
        _not = total_cmd._not
        shl = total_cmd.shl
        shr = total_cmd.shr
        sub = total_cmd.sub
        mov = total_cmd.mov
        ld_param = total_cmd.ld_param
        cvta = total_cmd.cvta
        setp = total_cmd.setp

        if fma > 0:
            temp1 = populate_alu_latencies("add") * (add + sub + mul + mad + _or + shl + shr + mov + cvta + setp + _and + _not)
            temp2 = populate_alu_latencies("global") * (st_global+ld_global) + populate_alu_latencies("shared") * (st_shared+ld_shared) + populate_alu_latencies("local") * (st_local + ld_local + ld_param)

            register_ = populate_resource()['register'] / registers / thread_per_block
            smem_ = populate_resource()['shared'] / smem
            '''score_1 = (temp2 + temp1) / fma / multiplier
            score_2 = (temp1 + temp2 + fma * populate_alu_latencies("fma")) / fma / multiplier
            score_3 = block_number / populate_resource()['sm']
            score_4 = thread_per_block / populate_resource()['cuda_cores']
            score_5 = thread_per_block / 32 if thread_per_block / 32 < 7 else 7
            #score = -(score_1 + score_2*1.5) * math.ceil(score_3) * math.ceil(score_4)
            #score = -(score_1 + score_2*1.5) * math.ceil(score_3) * score_4
            score = -(score_2) * math.ceil(score_3) * math.ceil(score_4) / score_5'''
            score_1 = (temp1 + temp2 + fma * populate_alu_latencies("fma")) / fma / multiplier
            score_2_temp = block_number / populate_resource()['sm']
            if score_2_temp <= 1 or score_2_temp < min(register_, smem_):
                score_2 = score_2_temp
                if score_2_temp > 1 and score_2_temp < min(register_, smem_):
                    score_2 = 1
            else:
                score_2 = 1 / (score_2_temp - min(register_, smem_))
            number_of_warps = thread_per_block / 32
            coeff = number_of_warps/4 if number_of_warps > 7 else 7/number_of_warps
            #score = -score_1 * math.ceil(score_2) * math.ceil(number_of_warps) * coeff
            score = -score_1 /  score_2 * math.ceil(number_of_warps) * coeff

        else:
            if (len(total_cmd_list) == 1):
                return -1000000
            # cost model for non-fma block
            temp1 = populate_alu_latencies("add") * (add + sub + mul + mad + _or + shl + shr + mov + cvta + setp + _and + _not + fma)
            temp2 = populate_alu_latencies("global") * (st_global+ld_global) + populate_alu_latencies("shared") * (st_shared+ld_shared) + populate_alu_latencies("local") * (st_local + ld_local + ld_param)

            score = -(temp1 + temp2) / multiplier
        # print("score: ", score)
        total_score += score
        if len(total_cmd_list) == 4:
            total_score = total_score / 10000
    return total_score

def populate_alu_latencies(op):
    latencies = {}
    latencies["add"]      = 4
    latencies["add_64"]   = 8
    latencies["sub"]      = 4
    latencies["mul"]      = 4
    latencies["mad"]      = 4
    latencies["and"]      = 4
    latencies["or"]       = 4 
    latencies["shl"]      = 4
    latencies["shr"]      = 4
    latencies["fma"]      = 4
    latencies["local"]    = 362
    latencies["global"]   = 362
    latencies["shared"]   = 18

    return latencies[op]

def populate_resource():
    resources = {}
    resources["shared"] = 49152 # Total amount of shared memory per block
    resources["register"] = 65536 # Total number of registers available per block
    resources["sm"] = 80 # Total number of streaming multiprocessor
    resources["cuda_cores"] = 64 # Total number of cuda cores per SM
    return resources


def get_gflops(r):
    cost = r[1].costs[0]
    wkl = r[0].task.workload
    cfg = r[0].config
    if "dense" in wkl[0]:
        m, k = wkl[1][1]
        n, _ = wkl[2][1]
        num_flop = 2 * k * m * n
    elif "batch_matmul" in wkl[0]:
        b, m, k = wkl[1][1]
        _, n, _ = wkl[2][1]
        num_flop = 2 * k * m * n * b

    elif "conv2d" in wkl[0]:
        n, ic, ih, iw = wkl[1][1]
        oc, _, kh, kw  = wkl[2][1]
        sh, sw = wkl[3]
        pl, pt, pr, pb = wkl[4]
        dh, dw = wkl[5]
        dilated_kernel_h = (kh - 1) * dh + 1
        dilated_kernel_w = (kw - 1) * dw + 1
        oh = (ih + pt + pb - dilated_kernel_h) // sh + 1
        ow = (iw + pl + pr - dilated_kernel_w) // sw + 1

        if "winograd" in wkl[0]:
            num_flop = 0

            tile_size = 4
            m = tile_size
            alpha = m + kw - 1
            nH, nW = (oh + m-1) // m, (ow + m-1) // m
            P = n * nH * nW

            VP = cfg['tile_p'].size[-1]
            VK = cfg['tile_k'].size[-1]

            # pack input tile
            num_flop += ic * P // VP * alpha * alpha * VP

            # transform image
            num_flop += ic * P // VP * alpha * alpha * VP * alpha * alpha * 2

            # batch gemm
            num_flop += alpha * alpha * oc * P * ic

            # inverse transform
            num_flop += oc * P * m * m * alpha * alpha * 2

            # unpack output
            num_flop += n * oc * oh * ow

        else:
            num_flop = 2 * n * oh * ow * oc * ic * kh * kw
    else:
        raise RuntimeError("Unsupported workload: {}.".format(wkl[0]))
    num_gflop = num_flop / 10e9
    fnum_flop = num_flop
    #if "winograd" in wkl[0]:
    #    fnum_flop *= 0.33

    if cost != 0:
        gflops = num_flop / 10e9 / cost
    else:
        gflops = None

    return fnum_flop, gflops