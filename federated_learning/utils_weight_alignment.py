from utils_libs import *
from utils_dataset import *
from utils_models import *
from tensorboardX import SummaryWriter
import pandas as pd

from utils_libs import *
from utils_dataset import *
from utils_models import *

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

import time

max_norm = 10


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0;
    loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval();
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions
def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node));
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].named_parameters()))

    param_0 = clnt_models[0].named_parameters()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train();
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, scheduler.get_lr()[0]))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                e + 1, acc_trn, loss_trn, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_weight_alignment(data_obj, act_prob, n_minibatch,
                learning_rate, batch_size, epoch, com_amount, print_per,
                weight_decay, model_func, init_model, alpha_coef,
                sch_step, sch_gamma, save_period,
                suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    suffix = f'Weight_alignment_{alpha_coef}_{suffix}'
    print(suffix)
    print(alpha_coef)

    sys.exit()
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_a%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, alpha_coef)
    suffix += '_seed%d' % rand_seed
    suffix += '_lrdecay%f' % lr_decay_per_round

    n_clnt = data_obj.n_client
    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list / np.sum(weight_list) * n_clnt
    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    avg_ins_mdls = list(range(n_save_instances))
    avg_all_mdls = list(range(n_save_instances))
    avg_cld_mdls = list(range(n_save_instances))

    trn_sel_clt_perf = np.zeros((com_amount, 2))
    tst_sel_clt_perf = np.zeros((com_amount, 2))

    trn_all_clt_perf = np.zeros((com_amount, 2))
    tst_all_clt_perf = np.zeros((com_amount, 2))

    trn_cur_cld_perf = np.zeros((com_amount, 2))
    tst_cur_cld_perf = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    clnt_models = list(range(n_clnt))
    saved_itr = -1

    ###
    state_gadient_diffs = np.zeros((n_clnt + 1, n_par)).astype('float32')  # including cloud state

    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt'
                              % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ####
                fed_ins = model_func()
                fed_ins.load_state_dict(
                    torch.load('%sModel/%s/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_ins.eval()
                fed_ins = fed_ins.to(device)

                for params in fed_ins.parameters():
                    params.requires_grad = False

                avg_ins_mdls[saved_itr // save_period] = fed_ins

                ####
                fed_all = model_func()
                fed_all.load_state_dict(
                    torch.load('%sModel/%s/%s/all_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_all.eval()
                fed_all = fed_all.to(device)

                # Freeze model
                for params in fed_all.parameters():
                    params.requires_grad = False

                avg_all_mdls[saved_itr // save_period] = fed_all

                ####
                fed_cld = model_func()
                fed_cld.load_state_dict(
                    torch.load('%sModel/%s/%s/cld_avg_%dcom.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_cld.eval()
                fed_cld = fed_cld.to(device)

                # Freeze model
                for params in fed_cld.parameters():
                    params.requires_grad = False

                avg_cld_mdls[saved_itr // save_period] = fed_cld

                if os.path.exists(
                        '%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_sel_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_all_clt_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_cur_cld_perf[:i + 1] = np.load(
                        '%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    parameter_drifts = np.load(
                        '%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, i + 1))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if (trial) or (
    not os.path.exists('%sModel/%s/%s/ins_avg_%dcom.pt' % (data_path, data_obj.name, suffix, com_amount))):

        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        else:
            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_cld.named_parameters())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            inc_seed = 0
            while (True):
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                unselected_clnts = np.sort(np.where(act_clients == False)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break

            global_mdl = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)  # Theta
            del clnt_models
            clnt_models = list(range(n_clnt))
            delta_g_sum = np.zeros(n_par)

            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                clnt_models[clnt] = model_func().to(device)
                model = clnt_models[clnt]
                model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
                for params in model.parameters():
                    params.requires_grad = True
                local_update_last = state_gadient_diffs[clnt]  # delta theta_i
                global_update_last = state_gadient_diffs[-1] / weight_list[clnt]  # delta theta
                alpha = alpha_coef / weight_list[clnt]
                hist_i = torch.tensor(parameter_drifts[clnt], dtype=torch.float32, device=device)  # h_i
                clnt_models[clnt] = train_weight_alignment(model, model_func, alpha, local_update_last, global_update_last,
                                                      global_mdl, hist_i,
                                                      trn_x, trn_y, learning_rate * (lr_decay_per_round ** i),
                                                      batch_size, epoch, print_per, weight_decay, data_obj.dataset,
                                                      sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                delta_param_curr = curr_model_par - cld_mdl_param
                parameter_drifts[clnt] += delta_param_curr
                beta = 1 / n_minibatch / learning_rate

                state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
                delta_g_cur = (state_g - state_gadient_diffs[clnt]) * weight_list[clnt]
                delta_g_sum += delta_g_cur
                state_gadient_diffs[clnt] = state_g
                clnt_params_list[clnt] = curr_model_par

            #new here
            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)
            global_model=model_func()
            print('here1')
            print(global_model)
            for i in range(len(clnt_params_list[selected_clnts])):
                model_i=set_client_from_params(model_func(), avg_mdl_param_sel)
                print("HERERERE")
                print(model_i)
            sys.exit()

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
            delta_g_cur = 1 / n_clnt * delta_g_sum
            state_gadient_diffs[-1] += delta_g_cur

            cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)

            avg_model_sel = set_client_from_params(model_func(), avg_mdl_param_sel)
            all_model = set_client_from_params(model_func(), np.mean(clnt_params_list, axis=0))

            cur_cld_model = set_client_from_params(model_func().to(device), cld_mdl_param)

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_sel_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_all_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            trn_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_sel_clt_perf[i][0],
                                   'All clients': trn_all_clt_perf[i][0],
                                   'Current cloud': trn_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_sel_clt_perf[i][1],
                                   'All clients': trn_all_clt_perf[i][1],
                                   'Current cloud': trn_cur_cld_perf[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model_sel, data_obj.dataset, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0],
                                   'Current cloud':
                                       get_acc_loss(cent_x, cent_y, cur_cld_model, data_obj.dataset, weight_decay)[0]
                               }, i
                               )

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model_sel, data_obj.dataset, 0)
            print("**** Cur Sel Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_sel_clt_perf[i] = [loss_tst, acc_tst]

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             all_model, data_obj.dataset, 0)
            print("**** Cur All Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_all_clt_perf[i] = [loss_tst, acc_tst]

            #####

            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))
            tst_cur_cld_perf[i] = [loss_tst, acc_tst]

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_sel_clt_perf[i][0],
                                   'All clients': tst_all_clt_perf[i][0],
                                   'Current cloud': tst_cur_cld_perf[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_sel_clt_perf[i][1],
                                   'All clients': tst_all_clt_perf[i][1],
                                   'Current cloud': tst_cur_cld_perf[i][1]
                               }, i
                               )

            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model_sel.state_dict(), '%sModel/%s/%s/ins_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/all_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(cur_cld_model.state_dict(), '%sModel/%s/%s/cld_avg_%dcom.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_sel_clt_perf[:i + 1])
                np.save('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_sel_clt_perf[:i + 1])
                np.save('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_all_clt_perf[:i + 1])
                np.save('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_all_clt_perf[:i + 1])

                np.save('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_cur_cld_perf[:i + 1])
                np.save('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_cur_cld_perf[:i + 1])

                # save parameter_drifts

                np.save('%sModel/%s/%s/%d_hist_params_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        parameter_drifts)
                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    # Delete the previous saved arrays
                    os.remove('%sModel/%s/%s/%d_com_trn_sel_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_sel_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_com_trn_all_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_all_clt_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_com_trn_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_com_tst_cur_cld_perf.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

                    os.remove('%sModel/%s/%s/%d_hist_params_diffs.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))
                    os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                    data_path, data_obj.name, suffix, i + 1 - save_period))

            if ((i + 1) % save_period == 0):
                avg_ins_mdls[i // save_period] = avg_model_sel
                avg_all_mdls[i // save_period] = all_model
                avg_cld_mdls[i // save_period] = cur_cld_model

    return avg_ins_mdls, avg_cld_mdls, avg_all_mdls, trn_sel_clt_perf, tst_sel_clt_perf, trn_cur_cld_perf, tst_cur_cld_perf, trn_all_clt_perf, tst_all_clt_perf
