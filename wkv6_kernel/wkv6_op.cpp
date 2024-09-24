/****************************************************************************************************
 * File				: wkv6_op.cpp
 * Date				: 2024-09-08 12:16:07
 * Author			: Eliwii_Keeya
 * Description		:
 * Last Modified	: 2024-09-08 12:16:07
 ****************************************************************************************************/
#include "kernel_operator.h"

class KernelWKV6
{
public:
    __aicore__ inline void KernelWKV6() {}
    __aicore__ inline void Init(GM_ADDR r, GM_ADDR k, float v, GM_ADDR w, GM_ADDR u, GM_ADDR workspace, GM_ADDR tiling)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->v = static_cast<float>(v)

        rGm.SetGlobalBuffer((__gm__ DTYPE_R *)r + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        kGm.SetGlobalBuffer((__gm__ DTYPE_K *)k + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        uGm.SetGlobalBuffer((__gm__ DTYPE_U *)u + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * sizeof(DTYPE_R));
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * sizeof(DTYPE_K));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(inQueueU, BUFFER_NUM, this->tileLength * sizeof(DTYPE_U));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));

        pipe.InitBuffer(bufQueueX, this->tileLength * sizeof(DTYPE_K));
        pipe.InitBuffer(bufQueueY, this->tileLength * sizeof(DTYPE_R));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_R> rLocal = inQueueR.AllocTensor<DTYPE_R>();
        AscendC::LocalTensor<DTYPE_K> kLocal = inQueueK.AllocTensor<DTYPE_K>();
        AscendC::LocalTensor<DTYPE_V> vLocal = inQueueV.AllocTensor<DTYPE_V>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_U> uLocal = inQueueU.AllocTensor<DTYPE_U>();
        AscendC::DataCopy(rLocal, rGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(kLocal, kGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(vLocal, vGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(wLocal, wGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(uLocal, uGm[progress * this->tileLength], this->tileLength);
        inQueueR.EnQue(rLocal);
        inQueueK.EnQue(kLocal);
        inQueueV.EnQue(vLocal);
        inQueueW.EnQue(wLocal);
        inQueueU.EnQue(uLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_R> rLocal = inQueueR.DeQue<DTYPE_R>();
        AscendC::LocalTensor<DTYPE_K> kLocal = inQueueK.DeQue<DTYPE_K>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_U> uLocal = inQueueU.DeQue<DTYPE_U>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_K> xTmp = bufQueueX.Get<DTYPE_K>();
        AscendC::LocalTensor<DTYPE_R> yTmp = bufQueueY.Get<DTYPE_R>();

        // forward r, k, v, w, u -> y
        AscendC::Muls(xTmp, kLocal, this->v, this->tileLength);
        AscendC::Mul(yTmp, uLocalm xTmp, this->tileLength);
        AscendC::Add(yTmp, yTmp, sLocal, this->tileLength);
        AscendC::Mul(yLocal, rLocal, yTmp, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueR.FreeTensor(rLocal);
        inQueueK.FreeTensor(kLocal);
        inQueueV.FreeTensor(vLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueU.FreeTensor(uLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    float v;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufQueueX, bufQueueY;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueR, inQueueK, inQueueW, inQueueU; 
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_R> rGm;
    AscendC::GlobalTensor<DTYPE_K> kGm;
    AscendC::GlobalTensor<DTYPE_W> wGm;
    AscendC::GlobalTensor<DTYPE_U> uGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void wkv6(int B, int T, int C, int H, GM_ADDR r, GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR u, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelWKV6 op;
    op.Init(r, k, v, w, u, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
