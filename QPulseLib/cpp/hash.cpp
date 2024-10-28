#include <hls_stream.h>
#include <ap_int.h>


#define PATTERNS 100   // wave conut in the library
#define TABLE_SIZE 400  // hashtable size, better larger than wave count

typedef ap_int<16> PID;
typedef struct {
    ap_int<16> key;
    ap_int<16> value;
} hash_in;

// init hashtable when updating circuit
void init_hash(hash_in hashtable[TABLE_SIZE], PID conv_rsts[PATTERNS]) {
#pragma HLS INLINE off

    for (int i = 0; i < TABLE_SIZE; i++) {
        hashtable[i].key = -1;
        hashtable[i].value = -1;
    }
    for (int j = 0; j < PATTERNS; j++) {
        int index = conv_rsts[j] % TABLE_SIZE;
        while (hashtable[index].key != -1) {
            index = (index + 1) % TABLE_SIZE;
        }
        hashtable[index].value = index;
    }
}



// key: conv_rst    value: index of the library, both pattern_lib and wave_lib
PID hash_search(PID conv_rst) {
#pragma HLS INLINE
    int index = conv_rst % TABLE_SIZE;

    // linear search and solve hash collision
    while (hashtable[index].key != pcode && hashtable[index].key != -1) {
        index = (index + 1) % TABLE_SIZE;
    }

    // find the exact value
    return hashtable[index].value;

}


/*
// �й�ϣ��ͻû���
void hash_update(Hash_update Pcodes_update[update_count]) {
    for (int i = 0; i < update_count; i++) {
        int index = Pcodes_update[i].old_pcode % WAVE_COUNT;
        while (hashtable[index].key != Pcodes_update[i].old_pcode) {
            index = (index + 1) % WAVE_COUNT;
        }
        hashtable[index].value = Pcode_update[i].new_pcode;
    }
}
*/


/*
hash_out hash_table_lookup(hash_in & in_key, bool & out_found, hash_out & out_value) {
    static Library table[TABLE_SIZE];

// #pragma HLS RESOURCE variable=table core=RAM_T2P_BRAM

    // ��ʼ����ϣ��
    for (int i = 0; i < TABLE_SIZE; i++) {
#pragma HLS PIPELINE II=1
        table[i].key = 0;
        table[i].value = 0;
    }
#pragma HLS PIPELINE
    // ���ú��ʵĹ�ϣ������������ɢ�洢
    for (int i = 0; i < TABLE_SIZE; i++) {
        ap_uint<32> key = in_key.read();
        ap_uint<10> index = key % TABLE_SIZE;
        table[index].key = key;
    }

    // �Թ�ϣ��������з�������֧�ָ��ߵĲ��ж�
#pragma HLS ARRAY_PARTITION variable=table complete

    while (!in_key.empty()) {
#pragma HLS PIPELINE II=1

        ap_uint<32> key = in_key.read();
        bool found = false;
        ap_uint<32> value = 0;

        // �����ϣ����
        ap_uint<10> index = key % TABLE_SIZE;

        // ��ϣ������
        for (int i = 0; i < TABLE_SIZE; i++) {
#pragma HLS UNROLL
            if (table[i].key == key) {
                found = true;
                value = table[i].value;
                break;
            }
        }

        // ������ҽ��
        out_found.write(found);
        out_value.write(value);
    }
}
*/