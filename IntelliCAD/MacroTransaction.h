/*
*	Copyright (C) 2019 APIless team. All right reserved.
*
*	���ϸ�			: MacroTransaction.h
*	�ۼ���			: �̼���
*	���� ������		: 19.03.23
*
*	���� ���̴� �Լ� �� ������ ��ũ�� �Լ� ���·� ����س��� ���
*/

#pragma once

/// <summary>
/// <para>�Լ��� ��ȯ ���� bool�� ���(ȣȯ ������ ��� bool Ÿ��)���� ����� ������ ��ũ�� �Լ��̴�.</para>
/// <para>���ڷ� ���� ���� true�̸� �ƹ��� �۾��� �������� �ʴ´�.</para>
/// <para>���ڷ� ���� ���� false�̸� ��� false�� ��ȯ�Ѵ�.</para>
/// </summary>
/// <param name="expr">�׽�Ʈ�� bool Ÿ�� ǥ����</param>
/// <returns>expr�� true�̸� ����. false�� ��쿡�� false ��ȯ</returns>
#define IF_F_RET_F(expr) if (!(expr)) return false

/// <summary>
/// <para>�Լ��� ��ȯ ���� bool�� ���(ȣȯ ������ ��� bool Ÿ��)���� ����� ������ ��ũ�� �Լ��̴�.</para>
/// <para>���ڷ� ���� ���� true�̸� ��� false�� ��ȯ�Ѵ�.</para>
/// <para>���ڷ� ���� ���� false�̸� �ƹ��� �۾��� �������� �ʴ´�.</para>
/// </summary>
/// <param name="expr">�׽�Ʈ�� bool Ÿ�� ǥ����</param>
/// <returns>expr�� true�̸� false ��ȯ. false�̸� ����</returns>
#define IF_T_RET_F(expr) if ((expr)) return false

/// <summary>
/// <para>�Լ��� ��ȯ ���� bool�� ���(ȣȯ ������ ��� bool Ÿ��)���� ����� ������ ��ũ�� �Լ��̴�.</para>
/// <para>���ڷ� ���� ���� true�̸� �ƹ��� �۾��� �������� �ʴ´�.</para>
/// <para>���ڷ� ���� ���� false�̸� ��� true�� ��ȯ�Ѵ�.</para>
/// </summary>
/// <param name="expr">�׽�Ʈ�� bool Ÿ�� ǥ����</param>
/// <returns>expr�� true�̸� ����. false�� ��쿡�� true ��ȯ</returns>
#define IF_F_RET_T(expr) if (!(expr)) return true

/// <summary>
/// <para>�Լ��� ��ȯ ���� bool�� ���(ȣȯ ������ ��� bool Ÿ��)���� ����� ������ ��ũ�� �Լ��̴�.</para>
/// <para>���ڷ� ���� ���� true�̸� ��� true�� ��ȯ�Ѵ�.</para>
/// <para>���ڷ� ���� ���� false�̸� �ƹ��� �۾��� �������� �ʴ´�.</para>
/// </summary>
/// <param name="expr">�׽�Ʈ�� bool Ÿ�� ǥ����</param>
/// <returns>expr�� true�̸� true ��ȯ. false�̸� ����</returns>
#define IF_T_RET_T(expr) if ((expr)) return true

/// <summary>
/// <para>�Լ��� ��ȯ ���� bool�� ���(ȣȯ ������ ��� bool Ÿ��)���� ����� ������ ��ũ�� �Լ��̴�.</para>
/// <para>���ڷ� ���� ���� true�̸� �ƹ��� �۾��� �������� �ʴ´�.</para>
/// <para>���ڷ� ���� ���� false�̸� ��� ��ȯ�Ѵ�.</para>
/// </summary>
/// <param name="expr">�׽�Ʈ�� bool Ÿ�� ǥ����</param>
#define IF_F_RET(expr) if (!(expr)) return

/// <summary>
/// <para>�Լ��� ��ȯ ���� bool�� ���(ȣȯ ������ ��� bool Ÿ��)���� ����� ������ ��ũ�� �Լ��̴�.</para>
/// <para>���ڷ� ���� ���� true�̸� ��� ��ȯ�Ѵ�.</para>
/// <para>���ڷ� ���� ���� false�̸� �ƹ��� �۾��� �������� �ʴ´�.</para>
/// </summary>
/// <param name="expr">�׽�Ʈ�� bool Ÿ�� ǥ����</param>
#define IF_T_RET(expr) if ((expr)) return