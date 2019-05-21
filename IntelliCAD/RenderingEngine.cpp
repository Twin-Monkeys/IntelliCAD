/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: RenderingEngine.cpp
*	�ۼ���			: ����
*	���� ������		: 19.04.07
*/

#include "RenderingEngine.h"
#include "RayBoxInstersector.h"
#include "NumberUtility.hpp"
#include "Constant.h"
#include "CudaHelper.h"
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
#include <device_launch_parameters.h>
#include <limits>

RenderingEngine RenderingEngine::__instance;

using namespace std;

namespace Device
{
	namespace Volume
	{
		/* variable */
		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� ���� ��Ÿ ������</para>
		/// <para>���� ũ��, ������ ����, ���� ���е��� ������.</para>
		/// </summary>
		__device__
		static VolumeMeta __meta[1];

		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� short Ÿ�� ���� 3���� �迭</para>
		/// <para>�����͸� 0.f~1.f ���̷� ����ȭ�ؼ� �����Ѵ�.</para>
		/// </summary>
		static cudaArray_t __pBuffer = nullptr;

		/// <summary>
		/// <para>GPU ���� 3���� �迭�� ���ε� �� �ؽ�ó �������̽�</para>
		/// <para>float Ÿ�� 3���� index�� �����Ͽ� �����Ǵ� ���� �����͸� ���� �����ؼ� �����´�.</para>
		/// <para>���� ����: tex3D(texMem, x, y, z);</para>
		/// </summary>
		static texture<ushort, 3, cudaReadModeNormalizedFloat> __texMem;

		/* function */
		/// <summary>
		/// GPU �޸𸮸� �Ҵ��ϰ�, �ؽ�ó �������̽��� ���ε� �Ѵ�.
		/// </summary>
		/// <param name="volumeData">
		/// <para>CPU ���� ������ ���۷���</para>
		/// <para>��Ÿ �����͸� �����Ѵ�.</para>
		/// </param>
		static void __malloc(const VolumeData& volumeData)
		{
			const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<ushort>();

			// ���� ��Ÿ �����͸� CPU���� GPU ������ �����Ѵ�.  
			cudaMemcpyToSymbol(__meta, &volumeData.meta, sizeof(VolumeMeta));
			const Size3D<size_t> VOL_SIZE = volumeData.meta.size.castTo<size_t>();

			// ���� 3���� �迭 �޸𸮸� GPU�� �Ҵ��Ѵ�.
			cudaMalloc3DArray(&__pBuffer, &DESC, { VOL_SIZE.width, VOL_SIZE.height, VOL_SIZE.depth });

			// �޸� ���� �Ķ���͸� �����Ѵ�.
			cudaMemcpy3DParms params = { 0 };
			params.srcPtr = make_cudaPitchedPtr(
				volumeData.pBuffer.get(), (VOL_SIZE.width * sizeof(ushort)), VOL_SIZE.width, VOL_SIZE.height);
			params.dstArray = __pBuffer;
			params.extent = { VOL_SIZE.width, VOL_SIZE.height, VOL_SIZE.depth };

			// ���� �����͸� CPU���� GPU �迭�� �����Ѵ�.
			cudaMemcpy3D(&params);

			__texMem.addressMode[0] = cudaAddressModeBorder;
			__texMem.addressMode[1] = cudaAddressModeBorder;
			__texMem.addressMode[2] = cudaAddressModeBorder;
			__texMem.filterMode = cudaFilterModeLinear;
			__texMem.normalized = false;

			// �ؽ�ó �����ڿ� ���� 3���� �迭�� ���ε��Ѵ�.
			cudaBindTextureToArray(__texMem, __pBuffer, DESC);
		}

		/// <summary>
		/// GPU �޸𸮸� ��ȯ�ϰ�, �ؽ�ó �������̽��� ���ε��� �����Ѵ�.
		/// </summary>
		static void __free()
		{
			if (__pBuffer)
			{
				cudaUnbindTexture(__texMem);
				cudaFreeArray(__pBuffer);

				__pBuffer = nullptr;
			}
		}
	}

	namespace TransferFunc
	{
		/* variable */
		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� float Ÿ�� 1���� red transfer function</para>
		/// <para>0.f~1.f ���� ���� ������.</para>
		/// </summary>
		static cudaArray_t __pRed = nullptr;

		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� float Ÿ�� 1���� green transfer function</para>
		/// <para>0.f~1.f ���� ���� ������.</para>
		/// </summary>
		static cudaArray_t __pGreen = nullptr;

		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� float Ÿ�� 1���� blue transfer function</para>
		/// <para>0.f~1.f ���� ���� ������.</para>
		/// </summary>
		static cudaArray_t __pBlue = nullptr;

		/// <summary>
		/// <para>GPU �޸𸮸� �Ҵ��ϴ� float Ÿ�� 1���� alpha transfer function</para>
		/// <para>0.f~1.f ���� ���� ������.</para>
		/// </summary>
		static cudaArray_t __pAlpha = nullptr;

		/// <summary>
		/// <para>GPU red ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� red ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float, 1, cudaReadModeElementType> __texMemRed;

		/// <summary>
		/// <para>GPU green ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� green ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float, 1, cudaReadModeElementType> __texMemGreen;

		/// <summary>
		/// <para>GPU blue ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� blue ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float, 1, cudaReadModeElementType> __texMemBlue;

		/// <summary>
		/// <para>GPU alpha ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� alpha ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float, 1, cudaReadModeElementType> __texMemAlpha;

		/* function */
		/// <summary>
		/// transfer function�� CPU���� GPU�� �����Ѵ�. 
		/// </summary>
		/// <param name="transferFunc">
		/// CPU trasfer function ���۷���
		/// </param>
		static void __memcpy(const TransferFunction& transferFunc)
		{
			const size_t SIZE =
				(sizeof(float) * static_cast<size_t>(transferFunc.PRECISION));

			cudaMemcpyToArray(__pRed, 0, 0, transferFunc.getRed(), SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pGreen, 0, 0, transferFunc.getGreen(), SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pBlue, 0, 0, transferFunc.getBlue(), SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pAlpha, 0, 0, transferFunc.getAlpha(), SIZE, cudaMemcpyHostToDevice);
		}

		/// <summary>
		/// GPU �޸𸮸� �Ҵ��ϰ�, �ؽ�ó �������̽��� ���ε� �Ѵ�.
		/// </summary>
		/// <param name="transferFunc">
		/// CPU trasfer function ���۷���
		/// </param>
		static void __malloc(const TransferFunction& transferFunc)
		{
			const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<float>();
			const size_t PRECISION = static_cast<size_t>(transferFunc.PRECISION);

			cudaMallocArray(&__pRed, &DESC, PRECISION, 1);
			cudaMallocArray(&__pGreen, &DESC, PRECISION, 1);
			cudaMallocArray(&__pBlue, &DESC, PRECISION, 1);
			cudaMallocArray(&__pAlpha, &DESC, PRECISION, 1);

			__memcpy(transferFunc);

			__texMemRed.normalized = true;
			__texMemRed.filterMode = cudaFilterModeLinear;
			__texMemRed.addressMode[0] = cudaAddressModeClamp;
			cudaBindTextureToArray(__texMemRed, __pRed, DESC);

			__texMemGreen.normalized = true;
			__texMemGreen.filterMode = cudaFilterModeLinear;
			__texMemGreen.addressMode[0] = cudaAddressModeClamp;
			cudaBindTextureToArray(__texMemGreen, __pGreen, DESC);

			__texMemBlue.normalized = true;
			__texMemBlue.filterMode = cudaFilterModeLinear;
			__texMemBlue.addressMode[0] = cudaAddressModeClamp;
			cudaBindTextureToArray(__texMemBlue, __pBlue, DESC);

			__texMemAlpha.normalized = true;
			__texMemAlpha.filterMode = cudaFilterModeLinear;
			__texMemAlpha.addressMode[0] = cudaAddressModeClamp;
			cudaBindTextureToArray(__texMemAlpha, __pAlpha, DESC);
		}

		/// <summary>
		/// GPU �޸𸮸� ��ȯ�ϰ�, �ؽ�ó �������̽��� ���ε��� �����Ѵ�.
		/// </summary>
		static void __free()
		{
			if (__pRed)
			{
				cudaUnbindTexture(__texMemRed);
				cudaFreeArray(__pRed);

				__pRed = nullptr;
			}

			if (__pGreen)
			{
				cudaUnbindTexture(__texMemGreen);
				cudaFreeArray(__pGreen);

				__pGreen = nullptr;
			}

			if (__pBlue)
			{
				cudaUnbindTexture(__texMemBlue);
				cudaFreeArray(__pBlue);

				__pBlue = nullptr;
			}

			if (__pAlpha)
			{
				cudaUnbindTexture(__texMemAlpha);
				cudaFreeArray(__pAlpha);

				__pAlpha = nullptr;
			}
		}
	}

	/* function */
	/// <summary>
	/// ��ũ���� ����, ���� �������� offset�� �̵��Ͽ� �ȼ��� ��ġ�� �˾Ƴ���.
	/// </summary>
	/// <param name="screenWidth">
	/// ��ũ�� �ʺ�
	/// </param>
	/// <param name="screenHeight">
	/// ��ũ�� ����
	/// </param>
	/// <param name="W_IDX">
	/// ���� ���� �ε���
	/// </param>
	/// <param name="H_IDX">
	/// ���� ���� �ε���
	/// </param>
	/// <param name="camPosition">
	/// ī�޶� ��ġ
	/// </param>
	/// <param name="orthoBasis">
	/// ���� ����
	/// </param>
	/// <param name="imgBasedSamplingStep">
	/// ��ũ�� offset �̵� ����
	/// </param>
	/// <returns>
	/// �ȼ� ��ġ
	/// </returns>
	__device__
	static Point3D __calcStartingPoint(
		const int screenWidth, const int screenHeight,
		const int wIdx, const int hIdx,
		const Point3D& camPosition, const OrthoBasis& orthoBasis,
		const float imgBasedSamplingStep)
	{
		const int SCR_WIDTH_HALF = (screenWidth / 2);
		const int SCR_HEIGHT_HALF = (screenHeight / 2);

		const float SCR_HORIZ_OFFSET = (static_cast<float>(wIdx - SCR_WIDTH_HALF) * imgBasedSamplingStep);
		const float SCR_VERT_OFFSET = (static_cast<float>(SCR_HEIGHT_HALF - hIdx) * imgBasedSamplingStep);

		Point3D samplePoint = camPosition;
		samplePoint += (orthoBasis.v * SCR_HORIZ_OFFSET);
		samplePoint += (orthoBasis.w * SCR_VERT_OFFSET);

		return samplePoint;
	}

	/// <summary>
	/// �־��� ���� ��ġ���� ���� ���͸� ����Ͽ� ��ȯ�Ѵ�.
	/// </summary>
	/// <param name="samplePoint">
	/// ���� ��ġ
	/// </param>
	/// <returns>
	/// ���� ����
	/// </returns>
	__device__ 
	static Vector3D __getNormal(const Point3D& samplePoint)
	{
		const float X_LEFT = tex3D(Volume::__texMem, (samplePoint.x - 1.f), samplePoint.y, samplePoint.z);
		const float X_RIGHT = tex3D(Volume::__texMem, (samplePoint.x + 1.f), samplePoint.y, samplePoint.z);

		const float Y_LEFT = tex3D(Volume::__texMem, samplePoint.x, (samplePoint.y - 1.f), samplePoint.z);
		const float Y_RIGHT = tex3D(Volume::__texMem, samplePoint.x, (samplePoint.y + 1.f), samplePoint.z);

		const float Z_LEFT = tex3D(Volume::__texMem, samplePoint.x, samplePoint.y, (samplePoint.z - 1.f));
		const float Z_RIGHT = tex3D(Volume::__texMem, samplePoint.x, samplePoint.y, (samplePoint.z + 1.f));

		return NumberUtility::inverseGradient(
			X_LEFT, X_RIGHT,
			Y_LEFT, Y_RIGHT,
			Z_LEFT, Z_RIGHT);
	}

	/// <summary>
	/// ������ ���� ������ �����Ѵ�.
	/// </summary>
	/// <param name="samplePoint">
	/// ���� ��ġ
	/// </param>
	/// <param name="INTENSITY">
	/// ���� ��ġ������ ���� ������ ��
	/// </param>
	/// <param name="alpha">
	/// ���� ��ġ������ alpha ��
	/// </param>
	/// <param name="shininess">
	/// ���� ����
	/// </param>
	/// <param name="light1">
	/// 1�� ���� ���۷���
	/// </param>
	/// <param name="light2">
	/// 2�� ���� ���۷���
	/// </param>
	/// <param name="light3">
	/// 3�� ���� ���۷���
	/// </param>
	/// <param name="camDirection">
	/// ī�޶� ���� ����
	/// </param>
	/// <returns></returns>
	__device__ static Color<float> __shade(
		const Point3D& samplePoint,
		const float intensity, const float alpha, const float shininess,
		const Light& light1, const Light& light2, const Light& light3, 
		const Vector3D& camDirection)
	{
		// ���� ������ ���� transfer function�� �־� albedo ������ �����´�.
		const float RED = tex1D(TransferFunc::__texMemRed, intensity);
		const float GREEN = tex1D(TransferFunc::__texMemGreen, intensity);
		const float BLUE = tex1D(TransferFunc::__texMemBlue, intensity);

		const Color<float> ALBEDO(RED, GREEN, BLUE);

		// ������ Ȱ��ȭ�Ǿ� ���� �ʴٸ� albedo�� �����Ѵ�.
		if (!light1.enabled && !light1.enabled && !light2.enabled)
			return ALBEDO;

		// ������ �ݿ��Ѵ�.
		const Light LIGHTS[3] = { light1, light1, light2 };
		Color<float> retVal(0.f);

		for (int i = 0; i < 3; ++i)
		{
			if (LIGHTS[i].enabled)
			{
				const Vector3D N = __getNormal(samplePoint);
				const Vector3D L = (LIGHTS[i].position - samplePoint).getUnit();
				const Vector3D V = -camDirection;
				const Vector3D H = ((L + V) / 2.f);

				// ambient
				retVal += (ALBEDO * LIGHTS[i].ambient);

				// diffuse
				float dotNL = N.dot(L);
				if (dotNL > 0.f)
					retVal += (ALBEDO * LIGHTS[i].diffuse * dotNL);

				// specular
				float dotNH = N.dot(H);
				if (dotNH > 0.f)
					retVal += (LIGHTS[i].specular * powf(dotNH, (shininess * 2.f)));
			}
		}

		return retVal;
	}

	/// <summary>
	/// �þ� ������ ������ �� ���� ����� ��ũ���� ����Ѵ�.
	/// </summary>
	/// <param name="screenWidth">
	/// ��ũ�� �ʺ�
	/// </param>
	/// <param name="screenHeight">
	/// ��ũ�� ����
	/// </param>
	/// <param name="camPosition">
	/// ī�޶� ��ġ
	/// </param>
	/// <param name="orthoBasis">
	/// ���� ����
	/// </param>
	/// <param name="light3">
	/// 1�� ����
	/// </param>
	/// <param name="light3">
	/// 2�� ����
	/// </param>
	/// <param name="light3">
	/// 3�� ����
	/// </param>
	/// <param name="shininess">
	/// ���� ����
	/// </param>
	/// <param name="imgBasedSamplingStep">
	/// ��ũ�� offset �̵� ����
	/// </param>
	/// <param name="objectBasedSamplingStep">
	/// �þ� ���� ���� ����
	/// </param>
	/// <param name="pScreen">
	/// ��ũ�� ������
	/// </param>
	__global__
	static void __raycast(
		const int screenWidth, const int screenHeight,
		const Point3D camPosition, const OrthoBasis orthoBasis,
		const Light light0, const Light light1, const Light light2, const float shininess,
		const float imgBasedSamplingStep, const float objectBasedSamplingStep, Pixel* pScreen)
	{
		// ��ũ���� �������� 2�����̳�, ���������δ� 1�����̴�.
		// ���� ��ũ�� �ȼ��� �����ϱ� ���ؼ���
		// ����, ���� ���� �ε����� ����Ͽ� offset�� ���ؾ� �Ѵ�.
		const int W_IDX = ((blockIdx.x * blockDim.x) + threadIdx.x);
		const int H_IDX = ((blockIdx.y * blockDim.y) + threadIdx.y);
		const int OFFSET = ((screenWidth * H_IDX) + W_IDX);

		// ������ �������� �ȼ� ��ġ�� �����Ѵ�.
		Point3D samplePoint = __calcStartingPoint(
			screenWidth, screenHeight, W_IDX, H_IDX, camPosition, orthoBasis, imgBasedSamplingStep);

		// �ȼ� ��ġ���� ī�޶� ���� �������� �þ� ������ �����Ͽ��� �� 
		// ������ �����ϴ��� ���θ� �����ϰ�, ���� ������ �˾Ƴ���.
		const Range<float> RANGE = RayBoxIntersector::getValidRange(
			Volume::__meta->size, samplePoint, orthoBasis.u);

		// �þ� ������ ������ �������� �ʴ´ٸ�, �ȼ� ���� 0���� �����ϰ� �����Ѵ�.
		if (RANGE.end < RANGE.start)
		{
			pScreen[OFFSET].set(0.f);
			return;
		}

		// ���� ��ġ�� ���� ���� ���� �������� �̵��Ѵ�.
		samplePoint += (orthoBasis.u * RANGE.start);

		float transparency = 1.f; // ���� ����
		Color<float> color(0.f); // ���� RGB ����

		for (float t = RANGE.start; t < RANGE.end; t += objectBasedSamplingStep)
		{
			// ���� ���� ��ġ���� ���� ������ ���� ���� �����Ͽ� �����´�.
			float INTENSITY = tex3D(Volume::__texMem, samplePoint.x, samplePoint.y, samplePoint.z);

			// ���� ������ ���� �����Ѵ�.
			const float CORRECTION_VALUE = (USHRT_MAX / Volume::__meta->voxelPrecision);
			const float CORRECTED_INTENSITY = (INTENSITY * CORRECTION_VALUE);

			// alpha transfer function�� ���� ���� �־� alpha ���� �˾Ƴ���.
			const float ALPHA = tex1D(TransferFunc::__texMemAlpha, CORRECTED_INTENSITY);

			// �þ� ���� ���� ������ ���� alpha ���� �����Ѵ�.
			// (���� ������ �۾����� alpha ���� �ݹ� �����Ǿ� ���ϴ� ����� ���� �� ����.)
			const float CORRECTED_ALPHA = (1.f - powf((1.f - ALPHA), objectBasedSamplingStep));

			// alpha�� 0�� �ƴ� ��쿡 ���ؼ�
			if (!NumberUtility::nearEqual(ALPHA, 0.f))
			{
				// shading ������ �����Ѵ�.
				const Color<float> SHADING_RESULT = __shade(
					samplePoint, CORRECTED_INTENSITY, CORRECTED_ALPHA, shininess, light0, light1, light2, orthoBasis.u);

				// alpha-blending
				const float VALUE = (transparency * CORRECTED_ALPHA);

				color.red += (VALUE * SHADING_RESULT.red);
				color.green += (VALUE * SHADING_RESULT.green);
				color.blue += (VALUE * SHADING_RESULT.blue);

				transparency *= (1.f - CORRECTED_ALPHA);

				// ���� ������ 0.01f���� �۾����� 
				// �þ� ������ ������ �����Ѵ�. (early-ray termination)
				if (transparency < 0.01f)
					break;
			}

			// �þ� ������ �����Ѵ�.
			samplePoint += orthoBasis.u;
		}

		// �ȼ� ���� alpha-blending compositing ���� ����� �����Ѵ�.
		pScreen[OFFSET].set(
			static_cast<ubyte>(NumberUtility::truncate(color.red * 255.f, 0.f, 255.f)),
			static_cast<ubyte>(NumberUtility::truncate(color.green * 255.f, 0.f, 255.f)),
			static_cast<ubyte>(NumberUtility::truncate(color.blue * 255.f, 0.f, 255.f)));
	}
}

////////////////////////////////
//// RENDERING ENGINE START ////
////////////////////////////////

/* constructor */
RenderingEngine::RenderingEngine() :
	volumeRenderer(__volumeMeta, __initialized), imageProcessor(__volumeMeta, __initialized)
{}

/* member function */
void RenderingEngine::loadVolume(const VolumeData& volumeData)
{
	__volumeMeta = volumeData.meta;

	Device::Volume::__free();
	Device::Volume::__malloc(volumeData);

	volumeRenderer.__onLoadVolume();
	imageProcessor.__onLoadVolume();

	__initialized = true;
}

void RenderingEngine::onLoadVolume(const VolumeData& volumeData)
{
	loadVolume(volumeData);
}

RenderingEngine& RenderingEngine::getInstance()
{
	return __instance;
}

///////////////////////////////
//// VOLUME RENDERER START ////
///////////////////////////////

/* constructor */
RenderingEngine::VolumeRenderer::VolumeRenderer(const VolumeMeta &volumeMeta, const bool &initFlag) :
	__volumeMeta(volumeMeta), __initialized(initFlag)
{

}

/* member function */
void RenderingEngine::VolumeRenderer::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight)
{
	if (__initialized)
	{
		const OrthoBasis ORTHOBASIS = camera.getOrthoBasis();

		Device::__raycast<<<dim3(screenWidth / 16, screenHeight / 16), dim3(16, 16)>>>(
			screenWidth, screenHeight,
			camera.getPosition(), ORTHOBASIS,
			__light[0], __light[1], __light[2], __shininess,
			__imgBasedSamplingStep, __objectBasedSamplingStep, pScreen);
	}
}

void RenderingEngine::VolumeRenderer::adjustImgBasedSamplingStep(const float delta) 
{
	__imgBasedSamplingStep += delta;

	if (__imgBasedSamplingStep < 0.1f)
		__imgBasedSamplingStep = 0.1f;
	else if (__imgBasedSamplingStep > 4.f)
		__imgBasedSamplingStep = 4.f;
}

void RenderingEngine::VolumeRenderer::__onLoadVolume()
{
	// ���ο� ������ ������ ������ ����� �� ���߿� ȣ��Ǵ� �ݹ� �Լ�
	// ���ο� ���� ���� �� �ʿ��� ó�� ��ƾ �ۼ�
	Device::TransferFunc::__free();
	__initTransferFunc();
	__initLight();
	__initCamera();
}

void RenderingEngine::VolumeRenderer::__initTransferFunc()
{
	__pTransferFunc = new TransferFunction(__volumeMeta.voxelPrecision);

	__pTransferFunc->setRed({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setGreen({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setBlue({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setAlpha({ 0, __volumeMeta.voxelPrecision });

	Device::TransferFunc::__malloc(*__pTransferFunc);
}

void RenderingEngine::VolumeRenderer::__initLight()
{
	__light[0].position = Constant::Light::Position::LEFT;
	__light[1].position = Constant::Light::Position::RIGHT;
	__light[2].position = Constant::Light::Position::BACK;

	__light[0].ambient = { 0.03f, 0.02f, 0.01f };
	__light[0].diffuse = { 0.6f, 0.1f, 0.1f };
	__light[0].specular = { 0.5f, 0.1f, 0.1f };

	__light[1].ambient = { 0.02f, 0.03f, 0.01f };
	__light[1].diffuse = { 0.1f, 0.6f, 0.1f };
	__light[1].specular = { 0.1f, 0.5f, 0.1f };

	__light[2].ambient = { 0.1f, 0.2f, 0.3f };
	__light[2].diffuse = { 0.2f, 0.4f, 0.6f };
	__light[2].specular = { 0.3f, 0.4f, 0.5f };
}

void RenderingEngine::VolumeRenderer::__initCamera()
{
	camera.set(Constant::Camera::EYE, Constant::Camera::AT, Constant::Camera::UP);
}

///////////////////////////////
//// IMAGE PROCESSOR START ////
///////////////////////////////

RenderingEngine::ImageProcessor::ImageProcessor(const VolumeMeta &volumeMeta, const bool &initFlag) :
	__volumeMeta(volumeMeta), __initialized(initFlag)
{

}

void RenderingEngine::ImageProcessor::__onLoadVolume()
{

}

void RenderingEngine::ImageProcessor::render(
	Pixel* const pScreen, const int screenWidth, const int screenHeight)
{
	
}