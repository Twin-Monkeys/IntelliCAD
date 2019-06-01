/*
*	Copyright (C) 2019 Jin Won. All right reserved.
*
*	���ϸ�			: RenderingEngine.cpp
*	�ۼ���			: ����
*	���� ������		: 19.04.07
*/

#include "custom_cudart.h"
#include "RenderingEngine.h"
#include "SystemIndirectAccessor.h"
#include "RayBoxInstersector.h"
#include "NumberUtility.hpp"
#include "Constant.h"
#include <limits>
#include "Index2D.hpp"
#include "VolumeIndicator.h"

RenderingEngine RenderingEngine::__instance;

using namespace std;

namespace Device
{
	static Light *__pLightsDev;

	// Volume Indicator
	__device__
	static float __indicatorLength[] = { 100.f };

	__device__
	static float __indicatorThickness[] = { 8.f };

	__device__
	static float __indicatorAlpha[] = { .4f };

	__device__
	static Color<float> __indicatorColor[] =
	{{ (220.f / 256.f), (25.f / 256.f), (72.f / 256.f) }};

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

			// ������ ũ��
			const cudaExtent VOL_SIZE =
				make_cudaExtent(volumeData.meta.size.width, volumeData.meta.size.height, volumeData.meta.size.depth);

			// ���� 3���� �迭 �޸𸮸� GPU�� �Ҵ��Ѵ�.
			cudaMalloc3DArray(&__pBuffer, &DESC, VOL_SIZE);

			// �޸� ���� �Ķ���͸� �����Ѵ�.
			cudaMemcpy3DParms params;
			ZeroMemory(&params, sizeof(cudaMemcpy3DParms));

			params.srcPtr = make_cudaPitchedPtr(
				volumeData.pBuffer.get(), (VOL_SIZE.width * sizeof(ushort)), VOL_SIZE.width, VOL_SIZE.height);
			
			params.dstArray = __pBuffer;
			params.extent = VOL_SIZE;

			// ���� �����͸� CPU���� GPU �迭�� �����Ѵ�.
			cudaMemcpy3D(&params);

			__texMem.addressMode[0] = cudaAddressModeBorder;
			__texMem.addressMode[1] = cudaAddressModeBorder;
			__texMem.addressMode[2] = cudaAddressModeBorder;
			__texMem.filterMode = cudaFilterModeLinear;

			// �ؽ�ó �����ڿ� ���� 3���� �迭�� ���ε��Ѵ�.
			cudaBindTextureToArray(__texMem, __pBuffer);
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
		static texture<float> __texMemRed;

		/// <summary>
		/// <para>GPU green ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� green ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float> __texMemGreen;

		/// <summary>
		/// <para>GPU blue ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� blue ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float> __texMemBlue;

		/// <summary>
		/// <para>GPU alpha ���Ϳ� ���ε� �� �ؽ�ó ������</para>
		/// <para>0.f~1.f ���̷� ����ȭ �� float Ÿ�� 1���� index�� ������ �� �ִ�.</para>
		/// <para>�̶� �����Ǵ� ���� alpha ���͸� ���� �����Ͽ� �����´�.</para>
		/// </summary>
		static texture<float> __texMemAlpha;

		/* function */
		/// <summary>
		/// transfer function�� CPU���� GPU�� �����Ѵ�. 
		/// </summary>
		/// <param name="transferFunc">
		/// CPU trasfer function ���۷���
		/// </param>
		static void __memcpy(const TransferFunction& transferFunc)
		{
			const size_t MEM_SIZE = (transferFunc.PRECISION * sizeof(float));

			cudaMemcpyToArray(__pRed, 0, 0, transferFunc.getRed(), MEM_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pGreen, 0, 0, transferFunc.getGreen(), MEM_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pBlue, 0, 0, transferFunc.getBlue(), MEM_SIZE, cudaMemcpyHostToDevice);
			cudaMemcpyToArray(__pAlpha, 0, 0, transferFunc.getAlpha(), MEM_SIZE, cudaMemcpyHostToDevice);
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
			const size_t MEM_SIZE = (transferFunc.PRECISION * sizeof(float));

			cudaMallocArray(&__pRed, &DESC, MEM_SIZE);
			cudaMallocArray(&__pGreen, &DESC, MEM_SIZE);
			cudaMallocArray(&__pBlue, &DESC, MEM_SIZE);
			cudaMallocArray(&__pAlpha, &DESC, MEM_SIZE);

			__memcpy(transferFunc);

			__texMemRed.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemRed, __pRed);

			__texMemGreen.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemGreen, __pGreen);

			__texMemBlue.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemBlue, __pBlue);

			__texMemAlpha.filterMode = cudaFilterModeLinear;
			cudaBindTextureToArray(__texMemAlpha, __pAlpha);
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
		const float SCR_WIDTH_HALF = (static_cast<float>(screenWidth) * .5f);
		const float SCR_HEIGHT_HALF = (static_cast<float>(screenHeight) * .5f);

		const float SCR_HORIZ_OFFSET = ((static_cast<float>(wIdx) - SCR_WIDTH_HALF) * imgBasedSamplingStep);
		const float SCR_VERT_OFFSET = ((SCR_HEIGHT_HALF - static_cast<float>(hIdx)) * imgBasedSamplingStep);

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
		const Point3D ADJ_POINT =
		{
			samplePoint.x + .5f,
			samplePoint.y + .5f,
			samplePoint.z + .5f
		};

		const float X_LEFT = tex3D(Volume::__texMem, (ADJ_POINT.x - 1.f), ADJ_POINT.y, ADJ_POINT.z);
		const float X_RIGHT = tex3D(Volume::__texMem, (ADJ_POINT.x + 1.f), ADJ_POINT.y, ADJ_POINT.z);

		const float Y_LEFT = tex3D(Volume::__texMem, ADJ_POINT.x, (ADJ_POINT.y - 1.f), ADJ_POINT.z);
		const float Y_RIGHT = tex3D(Volume::__texMem, ADJ_POINT.x, (ADJ_POINT.y + 1.f), ADJ_POINT.z);

		const float Z_LEFT = tex3D(Volume::__texMem, ADJ_POINT.x, ADJ_POINT.y, (ADJ_POINT.z - 1.f));
		const float Z_RIGHT = tex3D(Volume::__texMem, ADJ_POINT.x, ADJ_POINT.y, (ADJ_POINT.z + 1.f));

		return NumberUtility::inverseGradient(
			X_LEFT, X_RIGHT, Y_LEFT, Y_RIGHT, Z_LEFT, Z_RIGHT);
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
		const Light *const pLights, const Vector3D& camDirection)
	{
		// ���� ������ ���� transfer function�� �־� albedo ������ �����´�.
		const float RED = tex1D(TransferFunc::__texMemRed, intensity + .5f);
		const float GREEN = tex1D(TransferFunc::__texMemGreen, intensity + .5f);
		const float BLUE = tex1D(TransferFunc::__texMemBlue, intensity + .5f);

		const Color<float> ALBEDO(RED, GREEN, BLUE);

		// ������ Ȱ��ȭ�Ǿ� ���� �ʴٸ� albedo�� �����Ѵ�.
		if (!pLights[0].enabled && !pLights[1].enabled && !pLights[2].enabled)
			return ALBEDO;

		// ������ �ݿ��Ѵ�.
		Color<float> retVal(0.f);

		for (int i = 0; i < 3; ++i)
		{
			if (pLights[i].enabled)
			{
				const Vector3D N = __getNormal(samplePoint);
				const Vector3D L = (pLights[i].position - samplePoint).getUnit();
				const Vector3D V = -camDirection;
				const Vector3D H = ((L + V) * .5f);

				// ambient
				retVal += (ALBEDO * pLights[i].ambient);

				// diffuse
				float dotNL = N.dot(L);
				if (dotNL > 0.f)
					retVal += (ALBEDO * pLights[i].diffuse * dotNL);

				// specular
				float dotNH = N.dot(H);
				if (dotNH > 0.f)
					retVal += (pLights[i].specular * powf(dotNH, (shininess * 2.f)));
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
		const Light *const pLightsDev, const float shininess,
		const float imgBasedSamplingStep, const float objectBasedSamplingStep, Pixel* pScreen)
	{
		// ��ũ���� �������� 2�����̳�, ���������δ� 1�����̴�.
		// ���� ��ũ�� �ȼ��� �����ϱ� ���ؼ���
		// ����, ���� ���� �ε����� ����Ͽ� offset�� ���ؾ� �Ѵ�.
		const int W_IDX = ((blockIdx.x * blockDim.x) + threadIdx.x);
		const int H_IDX = ((blockIdx.y * blockDim.y) + threadIdx.y);
		const int OFFSET = ((screenWidth * H_IDX) + W_IDX);

		// ������ �������� �ȼ� ��ġ�� �����Ѵ�.
		const Point3D STARTING_POINT = __calcStartingPoint(
			screenWidth, screenHeight, W_IDX, H_IDX, camPosition, orthoBasis, imgBasedSamplingStep);

		// �ȼ� ��ġ���� ī�޶� ���� �������� �þ� ������ �����Ͽ��� �� 
		// ������ �����ϴ��� ���θ� �����ϰ�, ���� ������ �˾Ƴ���.
		const Range<float> RANGE = RayBoxIntersector::getValidRange(
			Volume::__meta->size, STARTING_POINT, orthoBasis.u);

		// �þ� ������ ������ �������� �ʴ´ٸ�, �ȼ� ���� 0���� �����ϰ� �����Ѵ�.
		if (RANGE.end < RANGE.start)
		{
			pScreen[OFFSET].set(0.f);
			return;
		}

		// ���� ��ġ�� ���� ���� ���� �������� �̵��Ѵ�.
		Point3D samplePoint = (STARTING_POINT + (orthoBasis.u * RANGE.start));

		float transparency = 1.f; // ���� ����
		Color<float> color(0.f); // ���� RGB ����

		const Vector3D STEP_VECTOR = (orthoBasis.u * objectBasedSamplingStep);

		// Volume indicator
		if (VolumeIndicator::recognize(
			Volume::__meta->size, samplePoint, *__indicatorLength, *__indicatorThickness))
		{
			color += (*__indicatorColor * *__indicatorAlpha);
			transparency = (1.f - *__indicatorAlpha);
		}

		for (float t = RANGE.start; t < RANGE.end; t += objectBasedSamplingStep)
		{
			// ���� ���� ��ġ���� ���� ������ ���� ���� �����Ͽ� �����´�.
			// 0.5�� ���ϴ� ���� CUDA�� �ؽ�ó �޸� Ư�� ����. ������ �� ��
			const float INTENSITY = tex3D(
				Volume::__texMem, samplePoint.x + .5f, samplePoint.y + .5f, samplePoint.z + .5f);

			// ���� ������ ���� �����Ѵ�.
			const float CORRECTED_INTENSITY = (INTENSITY * USHRT_MAX);

			// alpha transfer function�� ���� ���� �־� alpha ���� �˾Ƴ���.
			// ���� ����: ���� normalized �ε����� �ƴ�. ���� ���� ���� (RGB transfer function ��ο��� �ش��)
			const float ALPHA = tex1D(TransferFunc::__texMemAlpha, CORRECTED_INTENSITY + .5f);

			// �þ� ���� ���� ������ ���� alpha ���� �����Ѵ�.
			// (���� ������ �۾����� alpha ���� �ݹ� �����Ǿ� ���ϴ� ����� ���� �� ����.)
			const float CORRECTED_ALPHA = (1.f - powf((1.f - ALPHA), objectBasedSamplingStep));

			// alpha�� 0�� �ƴ� ��쿡 ���ؼ�
			if (!NumberUtility::nearEqual(ALPHA, 0.f))
			{
				// shading ������ �����Ѵ�.
				const Color<float> SHADING_RESULT = __shade(
					samplePoint, CORRECTED_INTENSITY, CORRECTED_ALPHA, shininess, pLightsDev, orthoBasis.u);

				// alpha-blending
				const float CUR_ALPHA = (transparency * CORRECTED_ALPHA);

				color.red += (CUR_ALPHA * SHADING_RESULT.red);
				color.green += (CUR_ALPHA * SHADING_RESULT.green);
				color.blue += (CUR_ALPHA * SHADING_RESULT.blue);

				transparency *= (1.f - CORRECTED_ALPHA);

				// ���� ������ 0.01f���� �۾����� 
				// �þ� ������ ������ �����Ѵ�. (early-ray termination)
				if (transparency < 0.01f)
					break;
			}

			// �þ� ������ �����Ѵ�.
			samplePoint += STEP_VECTOR;
		}

		// Volume indicator
		if (transparency > .01f)
		{
			samplePoint = (STARTING_POINT + (orthoBasis.u * RANGE.end));
			if (VolumeIndicator::recognize(
				Volume::__meta->size, samplePoint, *__indicatorLength, *__indicatorThickness))
				color += (*__indicatorColor * (transparency * *__indicatorAlpha));
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

void RenderingEngine::onSystemInit()
{
	SystemIndirectAccessor::getEventBroadcaster().addVolumeLoadingListener(*this);
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
	__volumeMeta(volumeMeta), __initialized(initFlag), camera(__imgBasedSamplingStep)
{
	cudaMalloc(&Device::__pLightsDev, 3 * sizeof(Light));
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
			Device::__pLightsDev, __shininess,
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

void RenderingEngine::VolumeRenderer::setIndicatorLength(const float length)
{
	cudaMemcpyToSymbol(Device::__indicatorLength, &length, sizeof(float));
}

void RenderingEngine::VolumeRenderer::setIndicatorThickness(const float thickness)
{
	cudaMemcpyToSymbol(Device::__indicatorThickness, &thickness, sizeof(float));
}

void RenderingEngine::VolumeRenderer::__onLoadVolume()
{
	// ���ο� ������ ������ ������ ����� �� ���߿� ȣ��Ǵ� �ݹ� �Լ�
	// ���ο� ���� ���� �� �ʿ��� ó�� ��ƾ �ۼ�
	Device::TransferFunc::__free();
	__initTransferFunc();
	__initLight();
	__initCamera();

	__imgBasedSamplingStep = 2.f;
	__objectBasedSamplingStep = 1.f;
}

void RenderingEngine::VolumeRenderer::__initTransferFunc()
{
	__pTransferFunc = new TransferFunction(__volumeMeta.voxelPrecision);

	/*__pTransferFunc->setRed({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setGreen({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setBlue({ 0, __volumeMeta.voxelPrecision });
	__pTransferFunc->setAlpha({ 0, __volumeMeta.voxelPrecision });*/

	ushort minVal = 3300;
	ushort maxVal = 4000;

	__pTransferFunc->setRed({ minVal, maxVal });
	__pTransferFunc->setGreen({ minVal, maxVal });
	__pTransferFunc->setBlue({ minVal, maxVal });
	__pTransferFunc->setAlpha({ minVal, maxVal });

	Device::TransferFunc::__malloc(*__pTransferFunc);
}

void RenderingEngine::VolumeRenderer::__initLight()
{
	__lights[0].position = Constant::Light::Position::LEFT;
	__lights[0].ambient = { 0.03f, 0.02f, 0.01f };
	__lights[0].diffuse = { 0.4f, 0.1f, 0.1f };
	__lights[0].specular = { 0.5f, 0.1f, 0.1f };

	__lights[1].position = Constant::Light::Position::RIGHT;
	__lights[1].ambient = { 0.02f, 0.03f, 0.01f };
	__lights[1].diffuse = { 0.1f, 0.6f, 0.1f };
	__lights[1].specular = { 0.1f, 0.5f, 0.1f };

	__lights[2].position = Constant::Light::Position::BACK;
	__lights[2].ambient = { 0.01f, 0.02f, 0.03f };
	__lights[2].diffuse = { 0.2f, 0.4f, 0.6f };
	__lights[2].specular = { 0.3f, 0.4f, 0.5f };

	cudaMemcpy(Device::__pLightsDev, __lights, 3 * sizeof(Light), cudaMemcpyKind::cudaMemcpyHostToDevice);
}

void RenderingEngine::VolumeRenderer::__initCamera()
{
	camera.set(Constant::Camera::EYE, Constant::Camera::AT, Constant::Camera::UP);
}

RenderingEngine::VolumeRenderer::~VolumeRenderer()
{
	cudaFree(Device::__pLightsDev);
}

///////////////////////////////
//// IMAGE PROCESSOR START ////
///////////////////////////////

namespace __Device
{
	namespace __ImgProc
	{
		texture<float> __transferTable;

		__device__
		Index2D<float> __calcSamplingPoint(
			const Index2D<> &screenIdx, const Size2D<> &screenSize,
			const float anchorHoriz, const float anchorVert, const float samplingStep)
		{
			const Size2D<float> SCR_SIZE_HALF = (screenSize.castTo<float>() / 2.f);

			const float SCR_HORIZ_OFFSET = ((static_cast<float>(screenIdx.x) - SCR_SIZE_HALF.width) * samplingStep);
			const float SCR_VERT_OFFSET = ((SCR_SIZE_HALF.height - static_cast<float>(screenIdx.y)) * samplingStep);

			Index2D<float> retVal = { anchorHoriz, anchorVert };

			retVal.x += SCR_HORIZ_OFFSET;
			retVal.y += SCR_VERT_OFFSET;

			return retVal;
		}

		__device__
		ubyte __shade(const Point3D &samplingPoint)
		{
			const float INTENSITY =
				(tex3D(Device::Volume::__texMem, samplingPoint.x + .5f, samplingPoint.y + .5f, samplingPoint.z + .5f) *
					static_cast<float>(USHRT_MAX));

			const float RET_VAL = tex1D(__transferTable, INTENSITY + .5f);

			return static_cast<ubyte>(NumberUtility::truncate(RET_VAL * 255.f, 0.f, 255.f));
		}

		__global__
		void __kernel_imgProcRender_top(
			Pixel* const pScreen, const Size2D<> screenSize, const Point2D anchor,
			const float samplingStep, const float volumeSlicingZ)
		{
			const Index2D<> SCR_IDX = Index2D<>::getKernelIndex();

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const Index2D<float> SAMPLING_PLANE =
				__calcSamplingPoint(SCR_IDX, screenSize, anchor.x, anchor.y, samplingStep);

			const ubyte RET_VAL = __shade({ SAMPLING_PLANE.x, SAMPLING_PLANE.y, volumeSlicingZ });
			
			pScreen[SCR_OFFSET].set(RET_VAL);
		}

		__global__
		void __kernel_imgProcRender_front(
			Pixel* const pScreen, const Size2D<> screenSize, const Point2D anchor,
			const float samplingStep, const float volumeSlicingY)
		{
			Index2D<> SCR_IDX = Index2D<>::getKernelIndex();
			SCR_IDX.y = ((screenSize.height - 1) - SCR_IDX.y);

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const Index2D<float> SAMPLING_PLANE =
				__calcSamplingPoint(SCR_IDX, screenSize, anchor.x, anchor.y, samplingStep);

			const ubyte RET_VAL = __shade({ SAMPLING_PLANE.x, volumeSlicingY, SAMPLING_PLANE.y });

			pScreen[SCR_OFFSET].set(RET_VAL);
		}

		__global__
		void __kernel_imgProcRender_right(
			Pixel* const pScreen, const Size2D<> screenSize, const Point2D anchor,
			const float samplingStep, const float volumeSlicingX)
		{
			using Device::Volume::__meta;

			const Index2D<> SCR_IDX = Index2D<>::getKernelIndex();

			const int TEX_Y = ((screenSize.height - 1) - SCR_IDX.y);
			const int SCR_OFFSET = ((TEX_Y * screenSize.width) + SCR_IDX.x);

			const Index2D<float> SAMPLING_PLANE =
				__calcSamplingPoint(SCR_IDX, screenSize, anchor.x, anchor.y, samplingStep);

			const ubyte RET_VAL = __shade({ volumeSlicingX, SAMPLING_PLANE.x, SAMPLING_PLANE.y });

			pScreen[SCR_OFFSET].set(RET_VAL);
		}
	}
}

RenderingEngine::ImageProcessor::ImageProcessor(const VolumeMeta &volumeMeta, const bool &initFlag) :
	__volumeMeta(volumeMeta), __initialized(initFlag),
	__slicingPointMgr(__samplingStep_top, __samplingStep_front, __samplingStep_right),
	__anchorMgr(__samplingStep_top, __samplingStep_front, __samplingStep_right)
{
	using namespace __Device::__ImgProc;
	
	__transferTable.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
}

void RenderingEngine::ImageProcessor::__onLoadVolume()
{
	__init();
}

void RenderingEngine::ImageProcessor::__sync()
{
	if (__transferTableDirty)
		__syncTransferFunction();
}

void RenderingEngine::ImageProcessor::__syncTransferFunction()
{
	const float RANGE_INV =
		(1.f / static_cast<float>(_transferTableBoundary.end - _transferTableBoundary.start));

	const Range<int> BOUNDARY = _transferTableBoundary.castTo<int>();

	const int ITER = __volumeMeta.voxelPrecision;
	for (int i = 0; i < ITER; i++)
	{
		if (i < BOUNDARY.start)
			__pHostTransferTableBuffer[i] = 0.f;
		else if (i < BOUNDARY.end)
			__pHostTransferTableBuffer[i] = (static_cast<float>(i - BOUNDARY.start) * RANGE_INV);
		else
			__pHostTransferTableBuffer[i] = 1.f;
	}

	cudaMemcpyToArray(
		__transferTableBuffer, 0, 0, __pHostTransferTableBuffer,
		__volumeMeta.voxelPrecision * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

	__transferTableDirty = false;
}

void RenderingEngine::ImageProcessor::__init()
{
	__release();

	__pHostTransferTableBuffer = new float[__volumeMeta.voxelPrecision];

	const cudaChannelFormatDesc DESC = cudaCreateChannelDesc<float>();

	cudaMallocArray(
		&__transferTableBuffer, &DESC, __volumeMeta.voxelPrecision * sizeof(float));

	using namespace __Device::__ImgProc;
	cudaBindTextureToArray(__transferTable, __transferTableBuffer);

	volumeHalfSize = (__volumeMeta.size.castTo<float>() * .5f);

	_transferTableBoundary.set(0, __volumeMeta.voxelPrecision);
	__transferTableDirty = true;

	__slicingPointMgr.init(__volumeMeta.size);
	__anchorMgr.init(__volumeMeta.size);

	__samplingStep_top = 2.f;
	__samplingStep_front = 2.f;
	__samplingStep_right = 2.f;
}

void RenderingEngine::ImageProcessor::__release()
{
	if (__transferTableBuffer)
	{
		using namespace __Device::__ImgProc;
		cudaUnbindTexture(__transferTable);

		cudaFreeArray(__transferTableBuffer);
		__transferTableBuffer = nullptr;

		delete[] __pHostTransferTableBuffer;
		__pHostTransferTableBuffer = nullptr;
	}
}

Index2D<> RenderingEngine::ImageProcessor::
getSlicingPointForScreen(const Size2D<> &screenSize, const SliceAxis axis)
{
	const Point3D &slicingPointAdj = __slicingPointMgr.getSlicingPointAdj();

	return __anchorMgr.getSlicingPointForScreen(screenSize, slicingPointAdj, axis);
}

void RenderingEngine::ImageProcessor::setTransferFunction(const ushort startInc, const ushort endExc)
{
	_transferTableBoundary.set(startInc, endExc);
	__transferTableDirty = true;
}

void RenderingEngine::ImageProcessor::setTransferFunction(const Range<ushort> &range)
{
	setTransferFunction(range.start, range.end);
}

void RenderingEngine::ImageProcessor::setSlicingPointFromScreen(
	const Size2D<> &screenSize, const Index2D<> &screenIdx, const SliceAxis axis)
{
	__slicingPointMgr.setSlicingPointFromScreen(
		screenSize, screenIdx, __anchorMgr.getAnchorAdj(axis), axis);
}

void RenderingEngine::ImageProcessor::adjustSlicingPoint(const float delta, const SliceAxis axis)
{
	__slicingPointMgr.adjustSlicingPoint(delta, axis);
}

void RenderingEngine::ImageProcessor::adjustSamplingStep(const float delta, const SliceAxis axis)
{
	float *pTarget = nullptr;

	switch (axis)
	{
	case SliceAxis::TOP:
		pTarget = &__samplingStep_top;
		break;

	case SliceAxis::FRONT:
		pTarget = &__samplingStep_front;
		break;

	case SliceAxis::RIGHT:
		pTarget = &__samplingStep_right;
		break;
	}

	*pTarget += delta;

	if (*pTarget < .1f)
		*pTarget = .1f;
	else if (*pTarget > 4.f)
		*pTarget = 4.f;
}

void RenderingEngine::ImageProcessor::adjustAnchor(
	const float deltaHoriz, const float deltaVert, const SliceAxis axis)
{
	__anchorMgr.adjustAnchor(deltaHoriz, deltaVert, axis);
}

void RenderingEngine::ImageProcessor::render(
	Pixel *const pScreen, const int screenWidth, const int screenHeight, const SliceAxis axis)
{
	if (__initialized)
	{
		__sync();

		using namespace __Device::__ImgProc;

		const dim3 gridDim =
		{
			static_cast<uint>(screenWidth / 16),
			static_cast<uint>(screenHeight / 16)
		};
		const dim3 blockDim = { 16, 16 };

		const Point3D &slicingPoint = __slicingPointMgr.getSlicingPoint();
		const Point2D &anchor = __anchorMgr.getAnchor(axis);

		switch (axis)
		{
		case SliceAxis::TOP:
			__kernel_imgProcRender_top<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, anchor, __samplingStep_top, slicingPoint.z);
			break;

		case SliceAxis::FRONT:
			__kernel_imgProcRender_front<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, anchor, __samplingStep_front, slicingPoint.y);
			break;

		case SliceAxis::RIGHT:
			__kernel_imgProcRender_right<<<gridDim, blockDim>>>(
				pScreen, { screenWidth, screenHeight }, anchor, __samplingStep_right, slicingPoint.x);
			break;
		}
	}
}

RenderingEngine::ImageProcessor::~ImageProcessor()
{
	__release();
}